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


def test_formal_names_are_the_only_public_runtime_vocabulary() -> None:
    """Plan、Session、Kernel、KernelStateだけを公開する。"""

    plan = cw.compile([cw.Flow([1])])
    session = plan.create_session()

    assert type(plan) is cw.Plan
    assert isinstance(session, cw.Session)
    for removed_name in (
        "ExecutionPlan",
        "ExecutionSession",
        "PlanSession",
        "PlanSessionState",
        "PlanSessionError",
        "CompiledKernel",
        "CompiledKernelSession",
        "BoundExecutionPlan",
        "NativeCompiledKernel",
        "NativeBatchCompiledKernel",
        "NativeBatchKernelSession",
    ):
        assert not hasattr(cw, removed_name)
    assert not hasattr(plan, "create_plan_session")
    assert not hasattr(plan, "create_continuous_session")


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


def test_repeated_session_run_recreates_kernel_state() -> None:
    """同じSessionのrun再実行でもKernelStateを共有しない。"""

    kernel = _LifecycleKernel()
    session = cw.compile([cw.Flow([1]).map(_LifecycleProvider(kernel))]).create_session()

    assert session.run().completed
    assert session.run().completed
    assert len(kernel.states) == 2
    assert kernel.states[0] is not kernel.states[1]
    assert [state.close_count for state in kernel.states] == [1, 1]


def test_session_flush_keeps_state_until_close() -> None:
    """flush後も同じKernelStateを保持し、close時に一度だけ解放する。"""

    kernel = _LifecycleKernel()
    mapped = cw.Flow([1]).map(_LifecycleProvider(kernel))
    session = cw.compile([mapped]).create_session()

    session.start()
    assert session.flush().completed
    assert len(kernel.states) == 1
    assert kernel.states[0].close_count == 0
    assert session.close().completed
    assert kernel.states[0].close_count == 1
    with pytest.raises(cw.SessionError, match="reusable run mode"):
        session.run()
