"""v0.3 Cython semantic prototype用の明示native契約を定義する。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import reduce
from math import isfinite
from operator import mul

from .kernel import CompileContext, CompiledKernel, RunContext
from .model import Emission, LogicalInterval, LogicalTime


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
        object.__setattr__(self, "items", tuple(normalized))
        object.__setattr__(self, "width", width)

    def __iter__(self) -> Iterator[VectorF64 | Emission[VectorF64]]:
        """正規化済みvectorまたはEmissionをSource順で返す。"""

        return iter(self.items)

    def emissions(self) -> tuple[Emission[VectorF64], ...]:
        """Cython境界用に全vectorを明示Emissionへ正規化する。"""

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
class _IdentityF64Session:
    def run(self, inputs: tuple[object, ...], context: RunContext) -> object:
        del context
        return inputs[0]


@dataclass(frozen=True)
class _CompiledIdentityF64:
    def create_session(self) -> _IdentityF64Session:
        return _IdentityF64Session()


@dataclass(frozen=True)
class IdentityF64Kernel:
    """f64値またはf64 frameを変更せず通すnative conformance Kernel。"""

    abi_version: str = "chronowire.kernel.identity_f64.v1"
    process_model: str = "identity_f64"

    def compile(self, context: CompileContext) -> CompiledKernel[object]:
        """Python基準実装用のstateless session factoryを返す。"""

        del context
        return _CompiledIdentityF64()


def identity_f64() -> IdentityF64Kernel:
    """Python/Cython同値性確認用identity f64 Kernelを返す。"""

    return IdentityF64Kernel()
