"""v0.3 Native Executor準備用PortablePlanIRを検証する。"""

import chronowire as cw


def test_v03_plan_records_stage_value_schema_and_experimental_kernel_abi() -> None:
    """Python依存とexecutor opcodeを推測不要なdescriptorとして記録する。"""

    source = cw.Flow([1, 2, 3, 4])
    framed = source.rate(4).frame(4)
    mapped = framed.map(sum)
    plan = cw.compile([cw.output(mapped, collector=cw.Latest())])

    ir = plan.portable_ir
    restored = cw.PortablePlanIR.from_json(ir.to_json())

    assert restored == ir
    assert ir.schema_version == "0.3"
    assert ir.value_schemas[0].representation == "python_opaque"
    assert [stage.execution_domain for stage in ir.stages] == ["python"]
    assert ir.stages[0].node_ids == (0, 1, 2, 3)
    assert ir.stages[0].boundary_reasons == (
        "python_island",
        "observation_boundary",
        "plan_end",
    )
    assert ir.stages[0].runner_capabilities == ("python_stage",)
    assert ir.stages[0].boundary_codec == "python_object"
    assert ir.kernel_abis[0].binding_slot == "kernel:3"
    assert ir.kernel_abis[0].process_model == "python_object"
    assert not ir.kernel_abis[0].native_compatible
    assert ir.kernel_abis[0].workspace_size_bytes is None
    assert ir.stream_item_abis == ()
    assert ir.native_buffers == ()


def test_v02_plan_without_native_descriptors_remains_readable() -> None:
    """schema 0.2 IRにv0.3 optional descriptorがなくても復元できる。"""

    payload = cw.compile([cw.Flow([1])]).portable_ir.to_dict()
    payload["schema_version"] = "0.2"
    del payload["value_schemas"]
    del payload["stages"]
    del payload["kernel_abis"]
    del payload["stream_item_abis"]
    del payload["native_buffers"]

    restored = cw.PortablePlanIR.from_dict(payload)

    assert restored.schema_version == "0.2"
    assert restored.value_schemas == ()
    assert restored.stages == ()
    assert restored.kernel_abis == ()
    assert restored.stream_item_abis == ()
    assert restored.native_buffers == ()
