"""リポジトリ外でbuildしたC ABI Operation moduleの動的bindingを検証する。"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

import chronowire as cw

_MODULE_SOURCE = r"""
#include "native_operation_abi.h"

#include <cstdio>
#include <new>

namespace {

int active_sessions = 0;
int created_sessions = 0;

void write_error(char* output, size_t capacity, const char* message) {
    if (output != nullptr && capacity > 0) {
        std::snprintf(output, capacity, "%s", message);
    }
}

void* create_scale(
    const double* parameters,
    size_t parameter_count,
    char* error_message,
    size_t error_capacity
) {
    if (parameters == nullptr || parameter_count != 1) {
        write_error(error_message, error_capacity, "scale requires one parameter");
        return nullptr;
    }
    double* session = new (std::nothrow) double(parameters[0]);
    if (session != nullptr) {
        ++active_sessions;
        ++created_sessions;
    }
    return session;
}

int process_scale(
    void* session,
    const CwBufferViewV1* inputs,
    size_t input_count,
    CwMutableBufferViewV1* output,
    CwProcessResultV1* result,
    char* error_message,
    size_t error_capacity
) {
    if (session == nullptr || inputs == nullptr || input_count != 1 || output == nullptr ||
        result == nullptr || inputs[0].value_count != output->value_capacity) {
        write_error(error_message, error_capacity, "scale buffer contract mismatch");
        return 1;
    }
    const double factor = *static_cast<double*>(session);
    for (size_t index = 0; index < inputs[0].value_count; ++index) {
        output->values[index] = inputs[0].values[index] * factor;
    }
    result->output_count = inputs[0].value_count;
    if (factor < 0.0) {
        result->status = 1;
        result->diagnostic_severity = 1;
        result->diagnostic_code = "NEGATIVE_SCALE";
        result->diagnostic_message = "external module applied a negative scale";
    }
    return 0;
}

void destroy_scale(void* session) {
    if (session != nullptr) {
        --active_sessions;
    }
    delete static_cast<double*>(session);
}

const CwOperationEntryV1 operations[] = {{
    sizeof(CwOperationEntryV1),
    "test.external_scale.v1",
    "test.external_scale.v1.cpp",
    "test.external-scale-abi.v1",
    "f64_item",
    sizeof(double),
    alignof(double),
    2,
    create_scale,
    process_scale,
    nullptr,
    destroy_scale,
}};

const CwOperationModuleV1 module = {
    sizeof(CwOperationModuleV1),
    CW_OPERATION_MODULE_ABI_V1,
    1,
    operations,
};

}  // namespace

extern "C" const CwOperationModuleV1* chronowire_operation_module_v1(void) {
    return &module;
}

extern "C" int chronowire_test_active_sessions(void) {
    return active_sessions;
}

extern "C" int chronowire_test_created_sessions(void) {
    return created_sessions;
}
"""


def _build_module(tmp_path: Path) -> Path:
    """一時directoryだけを使って共有library fixtureをbuildする。"""

    source = tmp_path / "scale_module.cpp"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library = tmp_path / f"libscale_module{suffix}"
    source.write_text(_MODULE_SOURCE, encoding="utf-8")
    include = cw.native_operation_include_dir()
    shared_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            shared_flag,
            "-fPIC",
            f"-I{include}",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


def test_native_operation_include_dir_exposes_packaged_wrapper_header() -> None:
    """DSP wrapperがChronowire内部pathを推測せず公開ABI headerを参照できる。"""

    include = cw.native_operation_include_dir()

    assert include.is_absolute()
    assert (include / "native_operation_abi.h").is_file()


def _operation() -> cw.OperationDefinition:
    """module manifestと同じ安定IDのOperationSpecを返す。"""

    return cw.declare_operation(
        operation_id="test.external_scale.v1",
        output="same",
        config=cw.ConfigSpec(scope="dsp.scale", fields={"factor": float}),
    )


def test_native_module_binds_without_serializing_library_path(tmp_path: Path) -> None:
    """module pathをIRへ入れずC++ runtimeでOperationを実行する。"""

    module = cw.NativeOperationModule(_build_module(tmp_path))
    backend = cw.NativeModuleBackend(module)
    config = cw.Config(dsp={"scale": {"factor": -2.0}})
    source = cw.f64_vector_source([(1.0, 2.0), (3.0, 4.0)], width=2)
    mapped = cw.Flow(source, config).map(_operation())
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))], backend=backend)

    cpp_result = plan.run(executor="cpp")
    python_result = plan.run(executor="python")
    assert python_result == cpp_result
    assert [item.value for item in cpp_result.outputs[0].emissions] == [
        (-2.0, -4.0),
        (-6.0, -8.0),
    ]
    assert all(
        item.status is cw.EmissionStatus.DEGRADED for item in cpp_result.outputs[0].emissions
    )
    assert all(
        item.diagnostics[0].code == "NEGATIVE_SCALE" for item in cpp_result.outputs[0].emissions
    )
    assert str(module.path) not in plan.portable_ir.to_json()


def test_native_module_python_executor_destroys_run_local_session(tmp_path: Path) -> None:
    """PythonExecutorの成功・Session closeでC ABI destroyを一度だけ呼ぶ。"""

    path = _build_module(tmp_path)
    module = cw.NativeOperationModule(path)
    library = ctypes.CDLL(str(path))
    active_sessions = library.chronowire_test_active_sessions
    active_sessions.argtypes = []
    active_sessions.restype = ctypes.c_int
    backend = cw.NativeModuleBackend(module)
    config = cw.Config(dsp={"scale": {"factor": 2.0}})
    mapped = cw.Flow(cw.f64_vector_source([(1.0, 2.0)], width=2), config).map(_operation())
    plan = cw.compile([cw.output(mapped, collector=cw.Latest())], backend=backend)

    plan.run(executor="python")
    assert active_sessions() == 0

    session = plan.create_session(executor="python")
    session.start()
    assert active_sessions() == 1
    session.close()
    assert active_sessions() == 0


def test_native_module_python_executor_recreates_session_at_gap(tmp_path: Path) -> None:
    """gap reset時に旧C ABI sessionを破棄してから新規生成する。"""

    path = _build_module(tmp_path)
    module = cw.NativeOperationModule(path)
    library = ctypes.CDLL(str(path))
    active_sessions = library.chronowire_test_active_sessions
    active_sessions.argtypes = []
    active_sessions.restype = ctypes.c_int
    created_sessions = library.chronowire_test_created_sessions
    created_sessions.argtypes = []
    created_sessions.restype = ctypes.c_int
    overrun = cw.Diagnostic(cw.Severity.WARNING, "INPUT_OVERRUN", "test gap")
    values = [
        cw.Emission(
            (1.0, 2.0),
            cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1)),
            0,
        ),
        cw.Emission(
            (3.0, 4.0),
            cw.LogicalInterval(cw.LogicalTime(2), cw.LogicalTime(3)),
            1,
            cw.EmissionStatus.DEGRADED,
            (overrun,),
        ),
    ]
    config = cw.Config(dsp={"scale": {"factor": 2.0}})
    mapped = cw.Flow(cw.f64_vector_source(values, width=2), config).map(_operation())
    plan = cw.compile(
        [cw.output(mapped, collector=cw.Bounded(2))],
        backend=cw.NativeModuleBackend(module),
    )

    plan.run(executor="python")

    assert created_sessions() == 2
    assert active_sessions() == 0


def test_operation_implementation_selection_is_independent_from_executor(
    tmp_path: Path,
) -> None:
    """OperationごとのPython/C++実装選択とPlan Executor選択を分離する。"""

    native = _operation()

    @cw.operation(
        operation_id="test.python_shift.v1",
        inputs={
            "input": cw.OperationInputSpec(
                primary=True,
                value=cw.ValueSpec(dtype="float64", shape=(2,)),
            )
        },
        output="same",
    )
    def shift(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        del config
        values = inputs["input"]
        assert isinstance(values, tuple)
        return tuple(float(value) + 1.0 for value in values)

    module = cw.NativeOperationModule(_build_module(tmp_path))
    native_backend = cw.NativeModuleBackend(module)
    config = cw.Config(dsp={"scale": {"factor": 2.0}})
    diagnostic = cw.Diagnostic(cw.Severity.WARNING, "MULTI_ISLAND_DEGRADED", "fallback")
    source = cw.Flow(
        cw.f64_vector_source(
            [
                cw.Emission(
                    (1.0, 2.0),
                    cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1)),
                    0,
                    cw.EmissionStatus.DEGRADED,
                    (diagnostic,),
                )
            ],
            width=2,
        ),
        config,
    )
    shifted = source.map(native).map(shift)
    plan = cw.compile(
        [cw.output(shifted, collector=cw.Latest())],
        backend="python",
        implementations={native.operation_id: native_backend},
    )

    result = plan.run(executor="python")

    assert result.outputs[0].emissions[0].value == (3.0, 5.0)
    assert plan.portable_ir.backend == "mixed"
    assert [stage.execution_domain for stage in plan.portable_ir.stages] == [
        "python_source",
        "cpp",
        "python",
    ]
    assert [stage.input_port_ids for stage in plan.portable_ir.stages] == [(), (0,), (1,)]
    assert [stage.output_port_ids for stage in plan.portable_ir.stages] == [(0,), (1,), (2,)]
    assert plan.portable_ir.stages[-1].boundary_codec == "stream_item_v1_to_python"
    assert [item.backend for item in plan.portable_ir.implementations] == ["cpp", "python"]
    cpp = plan.run(executor="cpp")

    assert cpp == result

    python_to_native = source.map(shift).map(native)
    unsupported = cw.compile(
        [cw.output(python_to_native, collector=cw.Latest())],
        backend="python",
        implementations={native.operation_id: native_backend},
    )
    assert unsupported.run(executor="cpp") == unsupported.run(executor="python")

    round_trip = source.map(native).map(shift).map(native)
    resumed = cw.compile(
        [cw.output(round_trip, collector=cw.Latest())],
        backend="python",
        implementations={native.operation_id: native_backend},
    )
    assert [stage.execution_domain for stage in resumed.portable_ir.stages] == [
        "python_source",
        "cpp",
        "python",
        "cpp",
    ]
    assert resumed.run(executor="cpp") == resumed.run(executor="python")

    two_islands = source.map(native).map(shift).map(native).map(shift)
    repeated = cw.compile(
        [cw.output(two_islands, collector=cw.Latest())],
        backend="python",
        implementations={native.operation_id: native_backend},
    )
    repeated_session = repeated.create_session(executor="cpp")
    repeated_result = repeated_session.run()

    assert isinstance(repeated_session, cw.Session)
    assert repeated_result == repeated.run(executor="python")
    assert repeated_session.last_metrics is not None
    assert repeated_session.last_metrics.execution_classification == "hybrid"
    assert repeated_session.last_metrics.stage_python_dispatches == 2
    assert repeated_session.last_metrics.gil_acquisitions == 2
    assert repeated_session.last_metrics.stage_boundary_batches == 3
    assert repeated_session.last_metrics.copied_batches == 3
    assert repeated_result.outputs[0].emissions[0].status is cw.EmissionStatus.DEGRADED
    assert repeated_result.outputs[0].emissions[0].diagnostics == (diagnostic,)

    should_fail = True

    def fail_once(value: object) -> object:
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("intentional multi-island failure")
        return value

    failing = source.map(native).map(shift).map(native).map(fail_once)
    retryable = cw.compile(
        [cw.output(failing, collector=cw.Latest())],
        backend="python",
        implementations={native.operation_id: native_backend},
    )
    with pytest.raises(cw.KernelExecutionError, match="intentional multi-island failure"):
        retryable.run(executor="cpp")
    assert retryable.run(executor="cpp").outputs[0].emissions


def test_cpp_executor_borrows_opted_in_fixed_schema_python_boundary(
    tmp_path: Path,
) -> None:
    """明示opt-inしたPython Operationの前後で同じread-only bufferを共有する。"""

    seen_views: list[memoryview] = []

    @cw.operation(
        operation_id="test.borrowed_identity.v1",
        inputs={
            "input": cw.OperationInputSpec(
                primary=True,
                value=cw.ValueSpec(dtype="float64", shape=(2,)),
            )
        },
        output="same",
        accepts_readonly_buffers=True,
    )
    def borrowed_identity(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        del config
        value = inputs["input"]
        if isinstance(value, memoryview):
            assert value.readonly
            assert value.format == "d"
            seen_views.append(value)
        return value

    native = _operation()
    backend = cw.NativeModuleBackend(cw.NativeOperationModule(_build_module(tmp_path)))
    source = cw.Flow(
        cw.f64_vector_source([(1.0, 2.0), (3.0, 4.0)], width=2),
        cw.Config(dsp={"scale": {"factor": 2.0}}),
    )
    flow = source.map(native).map(borrowed_identity).map(native)
    plan = cw.compile(
        [cw.output(flow, collector=cw.Bounded(2))],
        backend="python",
        implementations={native.operation_id: backend},
    )
    session = plan.create_session(executor="cpp")

    result = session.run()

    assert result == plan.run(executor="python")
    assert [item.value for item in result.outputs[0].emissions] == [
        (4.0, 8.0),
        (12.0, 16.0),
    ]
    assert len(seen_views) == 2
    assert seen_views[0].obj is seen_views[1].obj
    assert session.last_metrics is not None
    assert session.last_metrics.stage_boundary_batches == 2
    assert session.last_metrics.zero_copy_batches == 2
    assert session.last_metrics.copied_batches == 0
    first_owner = seen_views[0].obj
    seen_views.clear()

    assert session.run() == result
    assert len(seen_views) == 2
    assert seen_views[0].obj is not first_owner
    assert session.last_metrics is not None
    assert session.last_metrics.zero_copy_batches == 2
    implementation = next(
        item
        for item in plan.portable_ir.implementations
        if item.operation_id == borrowed_identity.operation_id
    )
    assert implementation.accepts_readonly_buffers
    round_tripped = cw.PortablePlanIR.from_json(plan.portable_ir.to_json())
    assert next(
        item
        for item in round_tripped.implementations
        if item.operation_id == borrowed_identity.operation_id
    ).accepts_readonly_buffers

    seen_views.clear()
    shared = source.map(native)
    fanout_plan = cw.compile(
        [
            cw.output(shared.map(borrowed_identity), collector=cw.Bounded(2)),
            cw.output(shared.map(borrowed_identity), collector=cw.Bounded(2)),
        ],
        backend="python",
        implementations={native.operation_id: backend},
    )
    fanout_session = fanout_plan.create_session(executor="cpp")

    assert fanout_session.run() == fanout_plan.run(executor="python")
    assert len(seen_views) == 4
    assert seen_views[0] is seen_views[2]
    assert seen_views[1] is seen_views[3]
    assert fanout_session.last_metrics is not None
    assert fanout_session.last_metrics.zero_copy_batches == 1
    assert fanout_session.last_metrics.copied_batches == 0

    @cw.operation(
        operation_id="test.borrowed_copy_fallback.v1",
        inputs={
            "input": cw.OperationInputSpec(
                primary=True,
                value=cw.ValueSpec(dtype="float64", shape=(2,)),
            )
        },
        output="same",
        accepts_readonly_buffers=True,
    )
    def copy_fallback(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        del config
        value = inputs["input"]
        assert isinstance(value, (tuple, memoryview))
        return tuple(float(item) + 1.0 for item in value)

    copied_plan = cw.compile(
        [cw.output(source.map(native).map(copy_fallback).map(native), collector=cw.Bounded(2))],
        backend="python",
        implementations={native.operation_id: backend},
    )
    copied_session = copied_plan.create_session(executor="cpp")

    assert copied_session.run() == copied_plan.run(executor="python")
    assert copied_session.last_metrics is not None
    assert copied_session.last_metrics.zero_copy_batches == 1
    assert copied_session.last_metrics.copied_batches == 1


@pytest.mark.parametrize("input_mode", ["synchronous", "latest"])
def test_cpp_executor_runs_multiple_input_python_stage_boundary(
    tmp_path: Path,
    input_mode: str,
) -> None:
    """native分岐から同期・latestの複数Portを一つのPython islandへ渡す。"""

    native = _operation()

    @cw.operation(
        operation_id=f"test.multi_input_{input_mode}.v1",
        inputs={
            "signal": cw.OperationInputSpec(
                primary=True,
                value=cw.ValueSpec(dtype="float64", shape=(2,)),
            ),
            "reference": cw.OperationInputSpec(
                mode=input_mode,
                value=cw.ValueSpec(dtype="float64", shape=(2,)),
            ),
        },
        output="same",
    )
    def combine(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        del config
        signal = inputs["signal"]
        reference = inputs["reference"]
        assert isinstance(signal, tuple)
        assert isinstance(reference, tuple)
        return tuple(
            float(signal_value) + float(reference_value)
            for signal_value, reference_value in zip(signal, reference, strict=True)
        )

    backend = cw.NativeModuleBackend(cw.NativeOperationModule(_build_module(tmp_path)))
    source = cw.Flow(
        cw.f64_vector_source([(1.0, 2.0), (3.0, 4.0)], width=2),
        cw.Config(dsp={"scale": {"factor": 2.0}}),
    )
    reference = source.map(native)
    signal = reference.map(native)
    reference_input: object = reference if input_mode == "synchronous" else reference.latest()
    combined = signal.map(combine, reference=reference_input)
    terminal_plan = cw.compile(
        [cw.output(combined, collector=cw.Bounded(2))],
        backend="python",
        implementations={native.operation_id: backend},
    )
    terminal_session = terminal_plan.create_session(executor="cpp")

    assert isinstance(terminal_session, cw.Session)
    assert [
        stage.input_port_ids
        for stage in terminal_plan.portable_ir.stages
        if stage.execution_domain == "python"
    ] == [(2, 1)]
    assert terminal_session.run() == terminal_plan.run(executor="python")
    assert terminal_session.last_metrics is not None
    assert terminal_session.last_metrics.stage_boundary_batches == 2
    assert terminal_session.last_metrics.copied_batches == 2

    # 二つ目のPython islandも置き、multi-island sessionが複数入力境界を扱うことを確認する。
    result_flow = combined.map(native).map(lambda value: value)
    plan = cw.compile(
        [cw.output(result_flow, collector=cw.Bounded(2))],
        backend="python",
        implementations={native.operation_id: backend},
    )

    python_result = plan.run(executor="python")
    cpp_session = plan.create_session(executor="cpp")
    cpp_result = cpp_session.run()

    assert cpp_result == python_result
    assert [item.value for item in cpp_result.outputs[0].emissions] == [
        (12.0, 24.0),
        (36.0, 48.0),
    ]
    assert isinstance(cpp_session, cw.Session)
    assert cpp_session.last_metrics is not None
    assert cpp_session.last_metrics.stage_python_dispatches == 2
    assert cpp_session.last_metrics.stage_boundary_batches == 4
    assert cpp_session.last_metrics.copied_batches == 4


def test_compile_rejects_unknown_operation_implementation_selector(tmp_path: Path) -> None:
    """使用しないOperation selectorを黙って無視しない。"""

    module = cw.NativeOperationModule(_build_module(tmp_path))
    mapped = cw.Flow([1]).map(lambda value: value)

    with pytest.raises(cw.CompileError, match="contract=known_implementation_selector"):
        cw.compile(
            [mapped],
            implementations={"test.unknown.v1": cw.NativeModuleBackend(module)},
        )


def test_cpp_python_prefix_preserves_zero_and_many_before_native_suffix(
    tmp_path: Path,
) -> None:
    """Python→native境界でSkipとEmitManyを一つのfixed-shape batchへpackする。"""

    calls = 0

    @cw.operation(
        operation_id="test.prefix_many.v1",
        inputs={
            "input": cw.OperationInputSpec(
                primary=True,
                value=cw.ValueSpec(dtype="float64", shape=(2,)),
            )
        },
        output=cw.OperationOutputSpec(value="same", emissions="many", max_items=2),
    )
    def expand(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        nonlocal calls
        del config
        calls += 1
        value = inputs["input"]
        if calls == 1:
            return cw.skip()
        return cw.emit_many((value, value))

    native = _operation()
    backend = cw.NativeModuleBackend(cw.NativeOperationModule(_build_module(tmp_path)))
    source = cw.Flow(
        cw.f64_vector_source([(1.0, 2.0), (3.0, 4.0)], width=2),
        cw.Config(dsp={"scale": {"factor": 2.0}}),
    )
    mapped = source.map(expand).map(native)
    plan = cw.compile(
        [cw.output(mapped, collector=cw.Bounded(2))],
        backend="python",
        implementations={native.operation_id: backend},
    )

    cpp = plan.run(executor="cpp")
    calls = 0

    assert cpp == plan.run(executor="python")
    assert [item.value for item in cpp.outputs[0].emissions] == [(6.0, 8.0)] * 2

    calls = 0
    repeated = source.map(native).map(expand).map(native).map(lambda value: value)
    multi_plan = cw.compile(
        [cw.output(repeated, collector=cw.Bounded(2))],
        backend="python",
        implementations={native.operation_id: backend},
    )

    multi_cpp = multi_plan.run(executor="cpp")
    calls = 0

    assert multi_cpp == multi_plan.run(executor="python")
    assert [item.value for item in multi_cpp.outputs[0].emissions] == [(12.0, 16.0)] * 2

    calls = 0
    prefix_repeated = source.map(expand).map(native).map(lambda value: value)
    prefix_multi_plan = cw.compile(
        [cw.output(prefix_repeated, collector=cw.Bounded(2))],
        backend="python",
        implementations={native.operation_id: backend},
    )

    prefix_multi_cpp = prefix_multi_plan.run(executor="cpp")
    calls = 0

    assert prefix_multi_cpp == prefix_multi_plan.run(executor="python")
    assert [item.value for item in prefix_multi_cpp.outputs[0].emissions] == [(6.0, 8.0)] * 2


def test_native_module_rebinds_round_tripped_plan(tmp_path: Path) -> None:
    """別process相当のIRをmodule ImplementationBindingから復元する。"""

    module = cw.NativeOperationModule(_build_module(tmp_path))
    backend = cw.NativeModuleBackend(module)
    config = cw.Config(dsp={"scale": {"factor": 3.0}})
    source = cw.f64_vector_source([(1.0, 2.0)], width=2)
    mapped = cw.Flow(source, config).map(_operation())
    original = cw.compile([cw.output(mapped, collector=cw.Bounded(1))], backend=backend)
    values: dict[str, object] = {}
    operation_id_by_node = {
        descriptor.node_id: descriptor.operation_id
        for descriptor in original.portable_ir.operations
    }
    for descriptor in original.portable_ir.bindings:
        if descriptor.kind == "source":
            values[descriptor.slot_id] = source
        elif descriptor.kind == "operation":
            assert descriptor.node_id is not None
            operation_id = operation_id_by_node[descriptor.node_id]
            values[descriptor.slot_id] = module.binding(operation_id)
        elif descriptor.kind == "collector":
            values[descriptor.slot_id] = cw.Bounded(1)

    rebound = cw.bind_plan(
        cw.PortablePlanIR.from_json(original.portable_ir.to_json()),
        cw.ExecutionBindings(values, {config.scope_id: config}),
        backend=backend,
    )

    assert rebound.run(executor="cpp").outputs[0].emissions[0].value == (3.0, 6.0)
