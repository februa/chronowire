"""v0.3 Cython semantic prototype用の明示native契約を定義する。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite

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
