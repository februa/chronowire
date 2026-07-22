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
    """PythonExecutorの成功・PlanSession closeでC ABI destroyを一度だけ呼ぶ。"""

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

    session = plan.create_plan_session(executor="python")
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
    source = cw.Flow(cw.f64_vector_source([(1.0, 2.0)], width=2), config)
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
    assert [item.backend for item in plan.portable_ir.implementations] == ["cpp", "python"]
    with pytest.raises(ValueError, match="contract=runtime_binding"):
        plan.run(executor="cpp")


def test_compile_rejects_unknown_operation_implementation_selector(tmp_path: Path) -> None:
    """使用しないOperation selectorを黙って無視しない。"""

    module = cw.NativeOperationModule(_build_module(tmp_path))
    mapped = cw.Flow([1]).map(lambda value: value)

    with pytest.raises(cw.CompileError, match="contract=known_implementation_selector"):
        cw.compile(
            [mapped],
            implementations={"test.unknown.v1": cw.NativeModuleBackend(module)},
        )


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
