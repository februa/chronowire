"""Plan/SessionとOperation/Kernel/KernelStateの正式命名契約を検証する。"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import chronowire as cw


class _LifecycleState:
    """processとcloseを持つ能動的なrun-local KernelState。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.close_count = 0

    def process(self, inputs: tuple[object, ...], context: cw.RunContext) -> object:
        del context
        if self.fail:
            raise RuntimeError("planned KernelState failure")
        return inputs[0]

    def close(self) -> None:
        self.close_count += 1


class _LifecycleKernel:
    """Sessionごとに独立したKernelStateを生成する不変Kernel。"""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.fail_first = fail_first
        self.states: list[_LifecycleState] = []

    def create_state(self) -> _LifecycleState:
        state = _LifecycleState(fail=self.fail_first and not self.states)
        self.states.append(state)
        return state


class _LifecycleProvider:
    """test用Kernelをcompile時に一度だけ返すlegacy provider。"""

    def __init__(self, kernel: _LifecycleKernel) -> None:
        self.kernel = kernel

    def compile(self, context: cw.CompileContext) -> _LifecycleKernel:
        del context
        return self.kernel


def test_formal_names_and_deprecated_aliases_reference_one_implementation() -> None:
    """互換aliasが内部実装を二重化せず正式型を参照する。"""

    plan = cw.compile([cw.Flow([1])])
    session = plan.create_session()

    assert type(plan) is cw.Plan
    assert isinstance(session, cw.Session)
    assert cw.ExecutionPlan is cw.Plan
    assert cw.ExecutionSession is cw.Session
    assert cw.PlanSession is cw.ContinuousSession
    assert cw.CompiledKernel is cw.Kernel
    assert cw.CompiledKernelSession is cw.KernelState


def test_deprecated_continuous_session_factory_warns() -> None:
    """旧methodは同じ実装へ委譲しつつ移行警告を出す。"""

    plan = cw.compile([cw.Flow([1])])

    with pytest.deprecated_call(match="create_continuous_session"):
        session = plan.create_plan_session()

    assert isinstance(session, cw.ContinuousSession)


def test_plan_is_structurally_immutable() -> None:
    """Planの属性とKernel mappingをcompile後に変更できない。"""

    plan = cw.compile([cw.Flow([1])])

    with pytest.raises(FrozenInstanceError):
        plan._backend_name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan._kernels[99] = _LifecycleKernel()  # type: ignore[index]


def test_kernel_state_is_closed_after_success_and_failure() -> None:
    """KernelStateを正常終了・例外終了の双方で破棄し、次Sessionへ共有しない。"""

    kernel = _LifecycleKernel(fail_first=True)
    mapped = cw.Flow([1]).map(_LifecycleProvider(kernel))
    plan = cw.compile([mapped])

    with pytest.raises(cw.KernelExecutionError, match="planned KernelState failure"):
        plan.run()
    assert kernel.states[0].close_count == 1

    assert plan.run().completed
    assert kernel.states[1].close_count == 1
    assert kernel.states[0] is not kernel.states[1]


def test_continuous_session_flush_keeps_state_until_close() -> None:
    """flush後も同じKernelStateを保持し、close時に一度だけ解放する。"""

    kernel = _LifecycleKernel()
    mapped = cw.Flow([1]).map(_LifecycleProvider(kernel))
    session = cw.compile([mapped]).create_continuous_session()

    session.start()
    assert session.flush().completed
    assert len(kernel.states) == 1
    assert kernel.states[0].close_count == 0
    assert session.close().completed
    assert kernel.states[0].close_count == 1
