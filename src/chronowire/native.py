"""v0.3 Cython semantic prototype用の明示native契約を定義する。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from math import isfinite

from .kernel import CompileContext, CompiledKernel, RunContext


@dataclass(frozen=True)
class F64SourceValues:
    """有限f64 scalar列として明示されたSource入力。

    Args:
        values: 有限な数値列。floatへ正規化して不変tupleに保持する。

    Raises:
        ValueError: boolまたはfloatへ変換できない値を含む場合。
    """

    values: tuple[float, ...]

    def __init__(self, values: Iterable[int | float]) -> None:
        normalized: list[float] = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError("f64 source values must not contain bool")
            try:
                converted = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("f64 source values must be finite numeric values") from error
            if not isfinite(converted):
                raise ValueError("f64 source values must be finite numeric values")
            normalized.append(converted)
        object.__setattr__(self, "values", tuple(normalized))

    def __iter__(self) -> Iterator[float]:
        """正規化済みf64値をSource順で返す。"""

        return iter(self.values)


def f64_source(values: Iterable[int | float]) -> F64SourceValues:
    """有限数値列を明示f64 Source入力へ変換する。

    Args:
        values: Source順の数値列。

    Returns:
        PythonExecutorとCythonExecutorの双方で使用できる不変入力。
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
