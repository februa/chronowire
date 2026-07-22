"""PortablePlanIRとprocess-local ExecutionBindingsの復元実行を検証する。"""

import pytest

import chronowire as cw


def _values(result: cw.RunResult) -> list[object]:
    return [item.value for item in result.outputs[0].emissions]


def _binding_values(
    plan: cw.Plan,
    *,
    source: object,
    kernel: object,
    collector: object,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for descriptor in plan.portable_ir.bindings:
        if descriptor.kind == "source":
            result[descriptor.slot_id] = source
        elif descriptor.kind == "kernel":
            result[descriptor.slot_id] = kernel
        elif descriptor.kind == "collector":
            result[descriptor.slot_id] = collector
    return result


def test_bind_plan_replays_v02_portable_ir_with_explicit_slots() -> None:
    """Python objectを含まないIRへ全process-local実体を明示注入する。"""

    values = [1, 2, 3, 4]
    operation = lambda frame: sum(frame)  # noqa: E731
    mapped = cw.Flow(values).frame(2).map(operation)
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))])
    restored_ir = cw.PortablePlanIR.from_json(plan.portable_ir.to_json())
    bindings = cw.ExecutionBindings(
        _binding_values(
            plan,
            source=values,
            kernel=operation,
            collector=cw.Bounded(2),
        )
    )

    rebound = cw.bind_plan(restored_ir, bindings)

    assert _values(rebound.run()) == [3, 7]
    assert rebound.portable_ir.schema_version == "0.3"


def test_bind_plan_reads_v01_descriptor_without_v02_optional_fields() -> None:
    """v0.1 IRのFRAME parameterをtime descriptorから復元する。"""

    values = [1, 2, 3, 4]
    operation = lambda frame: tuple(frame)  # noqa: E731
    mapped = cw.Flow(values).frame(2).map(operation)
    plan = cw.compile([cw.output(mapped, collector=cw.Bounded(2))])
    payload = plan.portable_ir.to_dict()
    payload["schema_version"] = "0.1"
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    assert isinstance(nodes, (list, tuple))
    assert isinstance(edges, (list, tuple))
    for node in nodes:
        if isinstance(node, dict):
            for key in ("frame_size", "frame_hop", "pad_end", "rate_period", "rate_policy"):
                node.pop(key, None)
    for edge in edges:
        if isinstance(edge, dict):
            edge.pop("tolerance", None)
            edge.pop("missing_policy", None)
    v01_ir = cw.PortablePlanIR.from_dict(payload)
    bindings = cw.ExecutionBindings(
        _binding_values(
            plan,
            source=values,
            kernel=operation,
            collector=cw.Bounded(2),
        )
    )

    assert _values(cw.bind_plan(v01_ir, bindings).run()) == [(1, 2), (3, 4)]


def test_bind_plan_rejects_missing_unknown_and_config_scope_mismatch() -> None:
    """slot集合とConfig scopeをGraph再構築前に明示検証する。"""

    config = cw.Config(scale=2)
    operation = lambda value, *, config: value * config.scale  # noqa: E731
    flow = cw.Flow([1], config).map(operation, config_paths=("scale",))
    plan = cw.compile([cw.output(flow, collector=cw.Latest())])
    values = _binding_values(
        plan,
        source=[1],
        kernel=operation,
        collector=cw.Latest(),
    )

    with pytest.raises(cw.ExecutionBindingError, match="missing=.*collector"):
        cw.bind_plan(
            plan.portable_ir,
            cw.ExecutionBindings(
                {key: value for key, value in values.items() if "collector" not in key}
            ),
        )
    with pytest.raises(cw.ExecutionBindingError, match="unknown=.*extra"):
        cw.bind_plan(
            plan.portable_ir,
            cw.ExecutionBindings({**values, "extra": object()}),
        )
    with pytest.raises(cw.ExecutionBindingError, match="missing Config binding"):
        cw.bind_plan(plan.portable_ir, cw.ExecutionBindings(values))

    rebound = cw.bind_plan(
        plan.portable_ir,
        cw.ExecutionBindings(values, configs={config.scope_id: config}),
    )
    assert _values(rebound.run()) == [2]
