"""v0.3 Cython semantic prototype用の明示native契約を定義する。"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from math import isfinite, lcm
from operator import mul

from .kernel import CompileContext, Kernel, NativeKernelRuntimeBinding, RunContext
from .model import Emission, EmissionStatus, LogicalInterval, LogicalTime

_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1
_STATUS_TO_NATIVE = {
    EmissionStatus.OK: 0,
    EmissionStatus.DEGRADED: 1,
    EmissionStatus.INVALID: 2,
}


def _normalized_f64(value: int | float) -> float:
    """boolと非有限値を除外してnative f64へ正規化する。"""

    if isinstance(value, bool):
        raise ValueError("f64 source values must not contain bool")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("f64 source values must be finite numeric values") from error
    if not isfinite(converted):
        raise ValueError("f64 source values must be finite numeric values")
    return converted


@dataclass(frozen=True)
class NativeValueBatch:
    """native Kernel境界を通るread-only contiguous f64 item batch。

    Args:
        values: native endian float64を連続格納したimmutable bytes。
        item_count: batch内の論理item数。
        item_shape: 一つのitemの固定shape。

    Raises:
        ValueError: 件数、shape、buffer byte数が契約と一致しない場合。
    """

    values: bytes
    item_count: int
    item_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.item_count < 0 or any(item <= 0 for item in self.item_shape):
            raise ValueError("native value batch count and shape must be valid")
        item_width = reduce(mul, self.item_shape, 1)
        if len(self.values) != self.item_count * item_width * 8:
            raise ValueError("native value batch byte length does not match its shape")

    def f64_view(self) -> memoryview[float]:
        """copyせずread-only float64一次元viewを返す。"""

        return memoryview(self.values).cast("d")


@dataclass(frozen=True)
class NativeF64Ingress:
    """CppExecutor sessionへbindする固定shape read-only f64 Source batch。

    Args:
        values: item-major contiguous native-endian float64。
        start_ticks: source timebase上のsigned i64 interval start列。
        end_ticks: source timebase上のsigned i64 interval end列。
        statuses: `OK=0, DEGRADED=1, INVALID=2`のu8列。
        resets: `INPUT_OVERRUN`直後にRATE/FRAME状態を破棄するu8列。
        item_count: Source item件数。
        width: 一つのitemのf64要素数。
        timebase_denominator: 一tickを`1 / denominator`論理秒とする正の分母。

    Raises:
        ValueError: byte長、件数、shapeまたはtimebaseが不正な場合。

    境界条件:
        Diagnostic本体はPython bindingが所有し、native runtimeはsource indexだけを伝播する。
    """

    values: bytes
    start_ticks: bytes
    end_ticks: bytes
    statuses: bytes
    resets: bytes
    item_count: int
    width: int
    timebase_denominator: int

    def __post_init__(self) -> None:
        if self.item_count < 0 or self.width <= 0 or self.timebase_denominator <= 0:
            raise ValueError("native ingress count, width, and timebase must be valid")
        if len(self.values) != self.item_count * self.width * 8:
            raise ValueError("native ingress value byte length does not match its shape")
        if len(self.start_ticks) != self.item_count * 8:
            raise ValueError("native ingress start tick byte length does not match item count")
        if len(self.end_ticks) != self.item_count * 8:
            raise ValueError("native ingress end tick byte length does not match item count")
        if len(self.statuses) != self.item_count:
            raise ValueError("native ingress status byte length does not match item count")
        if len(self.resets) != self.item_count or any(item > 1 for item in self.resets):
            raise ValueError("native ingress reset byte length or value is invalid")
        if any(item > 2 for item in self.statuses):
            raise ValueError("native ingress status is outside the StreamItem ABI")


def _checked_i64_ticks(value: Fraction, denominator: int) -> int:
    ticks = value * denominator
    if ticks.denominator != 1 or not _I64_MIN <= ticks.numerator <= _I64_MAX:
        raise ValueError("native ingress logical time is outside signed i64 tick range")
    return ticks.numerator


def _pack_vector_ingress(
    emissions: tuple[Emission[tuple[float, ...]], ...],
    width: int,
) -> NativeF64Ingress:
    denominator = 1
    for emission in emissions:
        denominator = lcm(
            denominator,
            emission.interval.start.as_fraction().denominator,
            emission.interval.end.as_fraction().denominator,
        )
        if denominator > _I64_MAX:
            raise ValueError("native ingress timebase exceeds signed i64 range")
    values = array("d", (value for emission in emissions for value in emission.value))
    starts = array(
        "q",
        (
            _checked_i64_ticks(emission.interval.start.as_fraction(), denominator)
            for emission in emissions
        ),
    )
    ends = array(
        "q",
        (
            _checked_i64_ticks(emission.interval.end.as_fraction(), denominator)
            for emission in emissions
        ),
    )
    if values.itemsize != 8 or starts.itemsize != 8 or ends.itemsize != 8:
        raise RuntimeError("native ingress requires 64-bit double and signed integer arrays")
    return NativeF64Ingress(
        values.tobytes(),
        starts.tobytes(),
        ends.tobytes(),
        bytes(_STATUS_TO_NATIVE[emission.status] for emission in emissions),
        bytes(
            any(diagnostic.code == "INPUT_OVERRUN" for diagnostic in emission.diagnostics)
            for emission in emissions
        ),
        len(emissions),
        width,
        denominator,
    )


VectorF64 = tuple[float, ...]


@dataclass(frozen=True)
class F64VectorSourceValues:
    """固定幅f64 vector列として明示されたSource入力。

    Args:
        values: 固定幅tupleまたはそのEmission列。
        width: vector要素数。

    Raises:
        ValueError: 幅、値、またはscalar/Emission形式が契約違反の場合。
    """

    items: tuple[VectorF64 | Emission[VectorF64], ...]
    width: int
    _emission_items: tuple[Emission[VectorF64], ...] = field(init=False, repr=False)
    _native_ingress: NativeF64Ingress = field(init=False, repr=False)

    def __init__(
        self,
        values: Iterable[tuple[int | float, ...] | Emission[tuple[int | float, ...]]],
        *,
        width: int,
    ) -> None:
        if isinstance(width, bool) or width <= 0:
            raise ValueError("f64 vector source width must be a positive integer")
        normalized: list[VectorF64 | Emission[VectorF64]] = []
        uses_emissions: bool | None = None
        for value in values:
            is_emission = isinstance(value, Emission)
            if uses_emissions is not None and uses_emissions != is_emission:
                raise ValueError("f64 vector source must not mix vectors and Emission values")
            uses_emissions = is_emission
            raw = value.value if is_emission else value
            if len(raw) != width:
                raise ValueError("f64 vector source item width does not match its contract")
            vector = tuple(_normalized_f64(item) for item in raw)
            if not is_emission:
                normalized.append(vector)
                continue
            normalized.append(
                Emission(
                    vector,
                    value.interval,
                    value.sequence,
                    value.status,
                    value.diagnostics,
                    value.metadata,
                )
            )
        normalized_items = tuple(normalized)
        emissions = tuple(
            item
            if isinstance(item, Emission)
            else Emission(
                item,
                LogicalInterval(LogicalTime(index), LogicalTime(index + 1)),
                index,
            )
            for index, item in enumerate(normalized_items)
        )
        object.__setattr__(self, "items", normalized_items)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "_emission_items", emissions)
        object.__setattr__(self, "_native_ingress", _pack_vector_ingress(emissions, width))

    def __iter__(self) -> Iterator[VectorF64 | Emission[VectorF64]]:
        """正規化済みvectorまたはEmissionをSource順で返す。"""

        return iter(self.items)

    def emissions(self) -> tuple[Emission[VectorF64], ...]:
        """全vectorを一度だけ正規化した不変Emission列として返す。"""

        return self._emission_items

    def native_ingress(self) -> NativeF64Ingress:
        """CppExecutorがPython値を再走査せずbindできるimmutable batchを返す。"""

        return self._native_ingress


def f64_vector_source(
    values: Iterable[tuple[int | float, ...] | Emission[tuple[int | float, ...]]],
    *,
    width: int,
) -> F64VectorSourceValues:
    """固定幅vector列を明示native f64 Sourceへ変換する。

    Args:
        values: Source順のvectorまたはEmission列。
        width: 一つのvectorの固定要素数。

    Returns:
        PythonExecutorとCythonExecutorの双方で使用できる不変入力。

    Raises:
        ValueError: vector shapeまたは値がnative契約を満たさない場合。
    """

    return F64VectorSourceValues(values, width=width)


@dataclass(frozen=True)
class F64SourceValues:
    """有限f64 scalar列として明示されたSource入力。

    Args:
        values: 有限な数値またはEmission列。値をfloatへ正規化して保持する。

    Raises:
        ValueError: bool、非有限値、変換不能値、形式の混在を含む場合。
    """

    items: tuple[float | Emission[float], ...]

    def __init__(self, values: Iterable[int | float | Emission[int | float]]) -> None:
        normalized: list[float | Emission[float]] = []
        uses_emissions: bool | None = None
        for value in values:
            is_emission = isinstance(value, Emission)
            if uses_emissions is not None and uses_emissions != is_emission:
                raise ValueError("f64 source must not mix scalar values and Emission values")
            uses_emissions = is_emission
            if not is_emission:
                normalized.append(_normalized_f64(value))
                continue
            normalized.append(
                Emission(
                    _normalized_f64(value.value),
                    value.interval,
                    value.sequence,
                    value.status,
                    value.diagnostics,
                    value.metadata,
                )
            )
        object.__setattr__(self, "items", tuple(normalized))

    @property
    def values(self) -> tuple[float, ...]:
        """値部分だけをSource順のf64 tupleとして返す。"""

        return tuple(item.value if isinstance(item, Emission) else item for item in self.items)

    def __iter__(self) -> Iterator[float | Emission[float]]:
        """正規化済みf64値またはEmissionをSource順で返す。"""

        return iter(self.items)

    def emissions(self) -> tuple[Emission[float], ...]:
        """Cython境界用に全itemを明示Emissionへ正規化する。"""

        return tuple(
            item
            if isinstance(item, Emission)
            else Emission(
                item,
                LogicalInterval(LogicalTime(index), LogicalTime(index + 1)),
                index,
            )
            for index, item in enumerate(self.items)
        )


def f64_source(
    values: Iterable[int | float | Emission[int | float]],
) -> F64SourceValues:
    """有限数値列を明示f64 Source入力へ変換する。

    Args:
        values: Source順の数値またはEmission列。両形式は混在不可。

    Returns:
        PythonExecutorとCythonExecutorの双方で使用できる不変入力。

    Raises:
        ValueError: native f64へ正規化できないか、値とEmissionを混在した場合。
    """

    return F64SourceValues(values)


@dataclass(frozen=True)
class _IdentityF64State:
    def process(self, inputs: tuple[object, ...], context: RunContext) -> object:
        del context
        return inputs[0]


@dataclass(frozen=True)
class _IdentityF64Kernel:
    def create_state(self) -> _IdentityF64State:
        return _IdentityF64State()

    def create_native_runtime_binding(self) -> NativeKernelRuntimeBinding:
        """parameterを持たないidentity ABI bindingを返す。"""

        return NativeKernelRuntimeBinding(
            "chronowire.kernel.identity_f64.v1",
            "identity_f64",
            "float64",
            (),
            b"",
        )


@dataclass(frozen=True)
class IdentityF64Kernel:
    """f64値またはf64 frameを変更せず通すnative conformance Kernel。"""

    abi_version: str = "chronowire.kernel.identity_f64.v1"
    process_model: str = "identity_f64"

    def compile(self, context: CompileContext) -> Kernel[object]:
        """Python基準実装用のstateless session factoryを返す。"""

        del context
        return _IdentityF64Kernel()


def identity_f64() -> IdentityF64Kernel:
    """Python/Cython同値性確認用identity f64 Kernelを返す。"""

    return IdentityF64Kernel()
