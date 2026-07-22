"""OperationSpec中心の公開宣言、compile、Python実行を検証する。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import pytest

import chronowire as cw


@runtime_checkable
class _HasShape(Protocol):
    """shape resolver testで実行時検証する最小schema境界。"""

    shape: tuple[int, ...] | None


def test_python_first_operation_receives_named_inputs_and_config_subtree() -> None:
    """decorator実装へ不変入力集合と選択ConfigViewだけを渡す。"""

    @cw.operation(
        operation_id="test.scale.v1",
        output="same",
        config=cw.ConfigSpec(scope="dsp.scale", fields={"factor": int}),
    )
    def scale(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        with pytest.raises(TypeError):
            inputs["input"] = 0  # type: ignore[index]
        value = inputs["input"]
        factor = config.factor
        assert isinstance(value, int)
        assert isinstance(factor, int)
        return value * factor

    mapped = cw.Flow(
        [1, 2],
        cw.Config(dsp={"scale": {"factor": 3}, "hidden": {"value": 9}}),
    ).map(scale)
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))])

    result = plan.run()

    assert [item.value for item in result.outputs[0].emissions] == [3, 6]
    abi = next(item for item in plan.portable_ir.kernel_abis if item.node_id == 1)
    assert abi.process_model == "python_operation"
    assert abi.abi_version == "chronowire.operation.python.v1"


def test_operation_binds_primary_synchronous_and_latest_inputs_by_name() -> None:
    """receiver、同期Flow、latest StateFlowを宣言名とmodeどおりにbindする。"""

    @cw.operation(
        operation_id="test.combine.v1",
        inputs={
            "signal": cw.OperationInputSpec(primary=True),
            "reference": cw.OperationInputSpec(),
            "gain": cw.OperationInputSpec(mode="latest"),
        },
        output="same",
    )
    def combine(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        del config
        signal_value = inputs["signal"]
        reference_value = inputs["reference"]
        gain_value = inputs["gain"]
        assert isinstance(signal_value, int)
        assert isinstance(reference_value, int)
        assert isinstance(gain_value, int)
        return signal_value + reference_value * gain_value

    signal = cw.Flow([1, 2])
    reference = signal.map(lambda value: value * 10)
    gain = signal.state_source([2])
    combined = signal.map(combine, reference=reference, gain=gain)

    result = cw.compile([cw.output(combined, collector=cw.Bounded(2))]).run()

    assert [item.value for item in result.outputs[0].emissions] == [21, 42]


def test_operation_rejects_unknown_missing_and_wrong_mode_inputs() -> None:
    """名前、必須性、Flow/StateFlow modeの違反をGraph構築時に明示する。"""

    declared = cw.declare_operation(
        operation_id="test.inputs.v1",
        inputs={
            "signal": cw.OperationInputSpec(primary=True),
            "state": cw.OperationInputSpec(mode="latest"),
        },
    )
    signal = cw.Flow([1])

    with pytest.raises(ValueError, match="requires input 'state'"):
        signal.map(declared)
    with pytest.raises(ValueError, match="unknown input 'extra'"):
        signal.map(declared, state=signal.latest(), extra=signal)
    with pytest.raises(ValueError, match="requires StateFlow"):
        signal.map(declared, state=signal)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (cw.Config(), "config_scope='dsp.fft' is unavailable"),
        (
            cw.Config(dsp={"fft": {"nfft": "1024"}}),
            "expected=int actual=str",
        ),
    ],
)
def test_operation_validates_config_scope_and_leaf_type(
    config: cw.Config,
    message: str,
) -> None:
    """OperationSpecが宣言したsubtree、leaf、型をcompile時に検証する。"""

    operation = cw.declare_operation(
        operation_id="test.fft.v1",
        output="same",
        config=cw.ConfigSpec(scope="dsp.fft", fields={"nfft": int}),
    )
    mapped = cw.Flow([1], config).map(operation)

    with pytest.raises(cw.MissingConfigError, match=message):
        cw.compile([mapped])


def test_operation_unifies_symbolic_input_shapes_before_backend_selection() -> None:
    """同名symbolの不一致をNode、Port、input、dimension付きで拒否する。"""

    operation = cw.declare_operation(
        operation_id="test.shape_merge.v1",
        inputs={
            "primary": cw.OperationInputSpec(
                primary=True,
                value=cw.ValueSpec(dtype="float64", shape=("samples", "channels")),
            ),
            "reference": cw.OperationInputSpec(
                value=cw.ValueSpec(dtype="float64", shape=("samples", "channels")),
            ),
        },
        output="same",
    )
    source = cw.Flow(cw.f64_vector_source([(1.0, 2.0)] * 6, width=2))
    primary = source.frame(2)
    reference = source.frame(3)
    merged = primary.map(operation, reference=reference)

    backend = _ShapeOnlyBackend()
    with pytest.raises(
        cw.ShapeMismatchError,
        match=r"operation=test\.shape_merge\.v1 input='reference'.*dimension=0",
    ):
        cw.compile([merged], backend=backend)
    assert backend.compile_count == 0


def test_operation_shape_resolver_records_resolved_output_schema() -> None:
    """compile-time resolverの結果だけをPort schemaへ記録する。"""

    def resolve_shape(
        inputs: Mapping[str, object],
        config: cw.ConfigView,
    ) -> tuple[int, ...]:
        del config
        schema = inputs["signal"]
        assert isinstance(schema, _HasShape)
        shape = schema.shape
        assert isinstance(shape, tuple)
        return (shape[0] * 2,)

    @cw.operation(
        operation_id="test.expand.v1",
        inputs={
            "signal": cw.OperationInputSpec(
                primary=True,
                value=cw.ValueSpec(dtype="float64", shape=("channels",)),
            )
        },
        output=cw.OperationOutputSpec(
            value=cw.ValueSpec(dtype="float64", shape=(None,)),
        ),
        shape_resolver=resolve_shape,
    )
    def expand(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        del config
        value = inputs["signal"]
        assert isinstance(value, tuple)
        return (*value, *value)

    mapped = cw.Flow(cw.f64_vector_source([(1.0, 2.0)], width=2)).map(expand)
    plan = cw.compile([cw.output(mapped, collector=cw.Latest())])
    output_port = next(item for item in plan.portable_ir.ports if item.port_id == mapped.port_id)
    schema = next(
        item
        for item in plan.portable_ir.value_schemas
        if item.value_schema_id == output_port.value_schema_id
    )

    assert schema.shape == (4,)
    assert schema.strides == (8,)
    assert plan.run().outputs[0].emissions[0].value == (1.0, 2.0, 1.0, 2.0)


def test_declared_operation_without_python_implementation_is_compile_error() -> None:
    """宣言のみOperationをPython Backendから黙って実行しない。"""

    declared = cw.declare_operation(operation_id="test.cpp_only.v1")
    mapped = cw.Flow([1]).map(declared)

    with pytest.raises(
        cw.MissingImplementationError,
        match=r"node=1 port=1 operation=test\.cpp_only\.v1 backend=python",
    ):
        cw.compile([mapped], backend="python")


class _ShapeOnlyState:
    def process(self, inputs: tuple[object, ...], context: cw.RunContext) -> object:
        del context
        return inputs[0]


class _ShapeOnlyKernel:
    def __init__(self, operation_id: str) -> None:
        self.implementation_spec = cw.ImplementationSpec(
            operation_id,
            f"{operation_id}.shape_only",
            "shape_only",
            "test.shape-only.v1",
        )

    def create_state(self) -> _ShapeOnlyState:
        return _ShapeOnlyState()


class _ShapeOnlyBackend:
    """宣言OperationをPython計算本体なしでcompileするtest Backend。"""

    name = "shape_only"

    def __init__(self) -> None:
        self.compile_count = 0

    def compile_operation(
        self,
        operation: cw.OperationSpec,
        context: object,
    ) -> cw.Kernel[object]:
        self.compile_count += 1
        del context
        return _ShapeOnlyKernel(operation.operation_id)

    def compile_kernel(
        self,
        kernel: object,
        context: cw.CompileContext,
    ) -> cw.Kernel[object]:
        del kernel, context
        raise AssertionError("legacy Kernel path must not be selected")


def test_cpp_first_backend_compiles_declaration_without_python_body() -> None:
    """外部Backendがoperation IDから実装を選びPython参照実装なしで実行する。"""

    declared = cw.declare_operation(operation_id="test.native_identity.v1")
    mapped = cw.Flow([1, 2]).map(declared)
    plan = cw.compile(
        [cw.output(mapped, collector=cw.Bounded(2))],
        backend=_ShapeOnlyBackend(),
    )

    assert [item.value for item in plan.run().outputs[0].emissions] == [1, 2]
    assert plan.portable_ir.backend == "shape_only"


def test_operation_multiple_outputs_keep_port_contracts_separate() -> None:
    """複数Portと一Port内Emission件数を混同せず配送する。"""

    @cw.operation(
        operation_id="test.split.v1",
        output={
            "positive": cw.OperationOutputSpec(value="same"),
            "negative": cw.OperationOutputSpec(value="same"),
        },
    )
    def split(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        del config
        value = inputs["input"]
        assert isinstance(value, int)
        return cw.kernel_outputs(value, -value)

    positive, negative = cw.Flow([1, 2]).map_outputs(split, output_count=2)
    plan = cw.compile(
        [
            cw.output(positive, collector=cw.Bounded(2)),
            cw.output(negative, collector=cw.Bounded(2)),
        ]
    )

    result = plan.run()
    assert [item.value for item in result.outputs[0].emissions] == [1, 2]
    assert [item.value for item in result.outputs[1].emissions] == [-1, -2]


def test_operation_preserves_degraded_and_skips_invalid_by_default() -> None:
    """安全な劣化情報を保持し、未受理INVALIDではPython実装を呼ばない。"""

    calls = 0

    @cw.operation(operation_id="test.status.v1", output="same")
    def status_operation(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        nonlocal calls
        del config
        calls += 1
        return inputs["input"]

    interval = cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1))
    diagnostic = cw.Diagnostic(cw.Severity.WARNING, "SAFE_FALLBACK", "fallback used")
    degraded = cw.Emission(
        1,
        interval,
        0,
        cw.EmissionStatus.DEGRADED,
        (diagnostic,),
    )
    invalid = cw.Emission(2, interval, 1, cw.EmissionStatus.INVALID)
    mapped = cw.Flow([degraded, invalid]).map(status_operation)

    emissions = cw.compile([cw.output(mapped, collector=cw.Bounded(2))]).run().outputs[0].emissions

    assert calls == 1
    assert emissions[0].status is cw.EmissionStatus.DEGRADED
    assert emissions[0].diagnostics == (diagnostic,)
    assert emissions[1].status is cw.EmissionStatus.INVALID
    assert emissions[1].diagnostics[-1].code == "INVALID_INPUT_PROPAGATED"


def test_operation_scalar_output_shape_and_missing_shape_config() -> None:
    """scalar shapeを固定schemaとして保持し、shape用Config欠落を明示する。"""

    scalar = cw.declare_operation(
        operation_id="test.scalar.v1",
        output=cw.OperationOutputSpec(value=cw.ValueSpec(dtype="float64", shape=())),
    )
    mapped = cw.Flow(cw.f64_source([1.0])).map(scalar)
    plan = cw.compile([mapped], backend=_ShapeOnlyBackend())
    port = next(item for item in plan.portable_ir.ports if item.port_id == mapped.port_id)
    schema = next(
        item
        for item in plan.portable_ir.value_schemas
        if item.value_schema_id == port.value_schema_id
    )
    assert schema.shape == ()

    configured = cw.declare_operation(
        operation_id="test.config_shape.v1",
        output=cw.OperationOutputSpec(
            value=cw.ValueSpec(dtype="float64", shape=("$config.channels",))
        ),
        config_scope="dsp",
    )
    configured_map = cw.Flow(cw.f64_source([1.0]), cw.Config(dsp={})).map(configured)
    with pytest.raises(cw.MissingConfigError, match="missing_shape_config"):
        cw.compile([configured_map], backend=_ShapeOnlyBackend())


def test_operation_descriptor_round_trip_contains_only_portable_contract() -> None:
    """schema 0.4へresolved意味論と実装IDだけを保存する。"""

    def resolve_shape(
        inputs: Mapping[str, object],
        config: cw.ConfigView,
    ) -> tuple[int, ...]:
        del inputs
        channels = config.channels
        assert isinstance(channels, int)
        return (channels,)

    @cw.operation(
        operation_id="test.portable.v1",
        inputs={
            "signal": cw.OperationInputSpec(
                primary=True,
                value=cw.ValueSpec(dtype="float64", shape=("channels",)),
            )
        },
        output=cw.OperationOutputSpec(value=cw.ValueSpec(dtype="float64", shape=(None,))),
        config=cw.ConfigSpec(scope="dsp", fields={"channels": int}),
        shape_resolver=resolve_shape,
    )
    def portable(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        del config
        return inputs["signal"]

    config = cw.Config(dsp={"channels": 2})
    mapped = cw.Flow(cw.f64_vector_source([(1.0, 2.0)], width=2), config).map(portable)
    ir = cw.compile([cw.output(mapped, collector=cw.Latest())]).portable_ir
    restored = cw.PortablePlanIR.from_json(ir.to_json())

    assert restored == ir
    assert ir.schema_version == "0.4"
    assert len(ir.operations) == 1
    assert len(ir.implementations) == 1
    operation_descriptor = ir.operations[0]
    implementation = ir.implementations[0]
    assert operation_descriptor.operation_id == "test.portable.v1"
    assert operation_descriptor.inputs[0].value_schema_id == "native:f64:2"
    assert operation_descriptor.outputs[0].value_schema_id.endswith("float64:2")
    assert operation_descriptor.config_scope_path == "dsp"
    assert operation_descriptor.config_fields[0].type_names == ("builtins.int",)
    assert operation_descriptor.binding_slot == "implementation:1"
    assert implementation.implementation_id == "test.portable.v1.python"
    assert implementation.binding_slot == operation_descriptor.binding_slot
    payload = ir.to_json()
    assert "resolve_shape" not in payload
    assert "python_binding" not in payload

    inconsistent = ir.to_dict()
    implementations = inconsistent["implementations"]
    assert isinstance(implementations, (list, tuple))
    implementations[0]["abi_version"] = "wrong-v1"
    with pytest.raises(ValueError, match="inconsistent implementation"):
        cw.PortablePlanIR.from_dict(inconsistent)


def test_operation_plan_rebinds_from_implementation_binding() -> None:
    """別process相当のIRをImplementationBindingとConfigから復元して実行する。"""

    @cw.operation(
        operation_id="test.rebind_scale.v1",
        output="same",
        config=cw.ConfigSpec(scope="dsp.scale", fields={"factor": int}),
    )
    def scale(inputs: Mapping[str, object], config: cw.ConfigView) -> object:
        value = inputs["input"]
        factor = config.factor
        assert isinstance(value, int)
        assert isinstance(factor, int)
        return value * factor

    config = cw.Config(dsp={"scale": {"factor": 4}})
    mapped = cw.Flow([1, 2], config).map(scale)
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))])
    operation_binding = scale.python_binding
    assert operation_binding is not None
    values: dict[str, object] = {}
    for descriptor in plan.portable_ir.bindings:
        if descriptor.kind == "source":
            values[descriptor.slot_id] = [1, 2]
        elif descriptor.kind == "operation":
            values[descriptor.slot_id] = operation_binding
        elif descriptor.kind == "collector":
            values[descriptor.slot_id] = cw.Bounded(2)

    rebound = cw.bind_plan(
        cw.PortablePlanIR.from_json(plan.portable_ir.to_json()),
        cw.ExecutionBindings(values, {config.scope_id: config}),
    )

    assert [item.value for item in rebound.run().outputs[0].emissions] == [4, 8]
    assert rebound.portable_ir.schema_version == "0.4"

    wrong = cw.ImplementationBinding(
        cw.ImplementationSpec(
            "test.rebind_scale.v1",
            "test.rebind_scale.v1.wrong",
            "python",
            "chronowire.operation.python.v1",
        ),
        operation_binding.implementation,
    )
    wrong_values = dict(values)
    wrong_values["implementation:1"] = wrong
    with pytest.raises(cw.ExecutionBindingError, match="implementation_identity"):
        cw.bind_plan(
            plan.portable_ir,
            cw.ExecutionBindings(wrong_values, {config.scope_id: config}),
        )
