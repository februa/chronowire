"""リポジトリ外でbuildしたC ABI Operation moduleの動的bindingを検証する。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import chronowire as cw

_MODULE_SOURCE = r"""
#include "native_operation_abi.h"

#include <cstdio>
#include <new>

namespace {

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
    return new (std::nothrow) double(parameters[0]);
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
"""


def _build_module(tmp_path: Path) -> Path:
    """一時directoryだけを使って共有library fixtureをbuildする。"""

    source = tmp_path / "scale_module.cpp"
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library = tmp_path / f"libscale_module{suffix}"
    source.write_text(_MODULE_SOURCE, encoding="utf-8")
    include = Path(cw.__file__).parent
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

    result = plan.run(executor="cpp")
    assert [item.value for item in result.outputs[0].emissions] == [(-2.0, -4.0), (-6.0, -8.0)]
    assert all(item.status is cw.EmissionStatus.DEGRADED for item in result.outputs[0].emissions)
    assert all(item.diagnostics[0].code == "NEGATIVE_SCALE" for item in result.outputs[0].emissions)
    assert str(module.path) not in plan.portable_ir.to_json()
    with pytest.raises(cw.KernelExecutionError, match="requires CppExecutor"):
        plan.run(executor="python")


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
