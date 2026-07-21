"""Kernel compile/run境界とPython Backendを検証する。"""

from __future__ import annotations

import chronowire as cw


class _CompiledScale:
    """compile済み係数からrun-local sessionを生成するtest Kernel。"""

    def __init__(self, scale: int) -> None:
        self._scale = scale

    def create_session(self) -> _ScaleSession:
        """状態を共有しないscale sessionを生成する。"""

        return _ScaleSession(self._scale)


class _ScaleSession:
    """一回のrunに閉じたscale処理を表す。"""

    def __init__(self, scale: int) -> None:
        self._scale = scale

    def run(self, inputs: tuple[object, ...], context: cw.RunContext) -> object:
        value = inputs[0]
        if not isinstance(value, int):
            raise TypeError("input must be int")
        return value * self._scale


class _StatefulCompiledCounter:
    """runごとにcounter=0から始まるsession factory。"""

    def create_session(self) -> _CounterSession:
        """空のcounter sessionを生成する。"""

        return _CounterSession()


class _CounterSession:
    """一回のrun内だけcounterを累積するsession。"""

    def __init__(self) -> None:
        self._count = 0

    def run(self, inputs: tuple[object, ...], context: cw.RunContext) -> object:
        self._count += 1
        return self._count


class _CounterKernel:
    """stateful session factoryを返すtest Kernel。"""

    def compile(self, context: cw.CompileContext) -> cw.CompiledKernel[object]:
        return _StatefulCompiledCounter()


class _ScaleKernel:
    """Configから係数を一回だけ解決するtest Kernel。"""

    def __init__(self) -> None:
        self.compile_count = 0

    def compile(self, context: cw.CompileContext) -> cw.CompiledKernel[object]:
        self.compile_count += 1
        scale = context.config.require("scale")
        if not isinstance(scale, int):
            raise TypeError("scale must be int")
        return _CompiledScale(scale)


class _RecordingBackend:
    """明示Kernelだけをcompileした回数を記録するtest Backend。"""

    name = "recording"

    def __init__(self) -> None:
        self.compile_count = 0

    def compile_kernel(
        self,
        kernel: cw.Kernel[object],
        context: cw.CompileContext,
    ) -> cw.CompiledKernel[object]:
        self.compile_count += 1
        return kernel.compile(context)


def test_kernel_compiles_once_and_plan_reuses_compiled_kernel() -> None:
    """Plan再実行時にKernel.compileを繰り返さないことを確認する。"""

    kernel = _ScaleKernel()
    mapped = cw.Flow([1, 2], cw.Config(scale=3)).map(kernel)
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))])

    assert kernel.compile_count == 1
    assert [item.value for item in plan.run().outputs[0].emissions] == [3, 6]
    assert [item.value for item in plan.run().outputs[0].emissions] == [3, 6]
    assert kernel.compile_count == 1


def test_unknown_backend_is_rejected_at_compile() -> None:
    """未実装Backendへ暗黙fallbackせず、指定誤りをcompile時に検出する。"""

    mapped = cw.Flow([1]).map(lambda value: value)
    try:
        cw.compile([mapped], backend="missing")
    except ValueError as error:
        assert "unsupported backend" in str(error)
    else:
        raise AssertionError("unknown backend must be rejected")


def test_compiled_kernel_session_is_recreated_for_each_run() -> None:
    """CompiledKernelの可変状態が別runへ持ち越されないことを確認する。"""

    mapped = cw.Flow([10, 20]).map(_CounterKernel())
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))])

    assert [item.value for item in plan.run().outputs[0].emissions] == [1, 2]
    assert [item.value for item in plan.run().outputs[0].emissions] == [1, 2]


def test_python_callable_uses_python_domain_with_custom_backend() -> None:
    """Python callbackを選択Backendへ渡さず、明示的なPython境界として残す。"""

    backend = _RecordingBackend()
    python_mapped = cw.Flow([1], cw.Config(scale=2)).map(lambda value: value + 1)
    native_candidate = python_mapped.map(_ScaleKernel())
    plan = cw.compile(
        [cw.output(native_candidate, collector=cw.Latest())],
        backend=backend,
    )

    assert backend.compile_count == 1
    assert plan.run().outputs[0].emissions[0].value == 4
