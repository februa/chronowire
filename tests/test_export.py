"""Logical GraphとExecutionPlanのexport契約を検証する。"""

import json
from pathlib import Path

import chronowire as cw


def test_graph_info_and_export_include_edges(tmp_path: Path) -> None:
    """Flow引数によるデータ移動がGraphInfoとJSONに残ることを確認する。"""

    source = cw.Flow([1, 2])
    reference = source.map(lambda value: value * 10)
    combined = source.map(lambda value, *, ref: value + ref, ref=reference)
    info = combined.graph_info()
    path = tmp_path / "graph.json"
    combined.export(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(info.nodes) == 3
    assert len(info.edges) == 3
    assert len(payload["edges"]) == 3
    assert payload["edges"][-1]["keyword"] == "ref"


def test_plan_export_contains_outputs_and_compile_warning(tmp_path: Path) -> None:
    """Plan JSONがcollectorとcompile warningを再現可能に記録する。"""

    source = cw.Flow([1, 2, 3])
    framed = source.frame(2)
    merged = source.map(lambda value, *, frame: (value, frame), frame=framed)
    plan = cw.compile([cw.output(merged, collector=cw.Latest())])
    json_path = tmp_path / "plan.json"
    dot_path = tmp_path / "plan.dot"

    plan.export(json_path)
    plan.export(dot_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["outputs"][0]["collector_kind"] == "latest"
    assert payload["diagnostics"][0]["code"] == "POSSIBLE_INTERVAL_MISMATCH"
    assert payload["nodes"][-1]["execution_domain"] == "python"
    assert payload["buffers"][0]["reclaim_policy"] == "all_consumers_advanced"
    assert "digraph chronowire_plan" in dot_path.read_text(encoding="utf-8")


def test_rate_period_is_exported_in_graph_and_plan(tmp_path: Path) -> None:
    """rate周期とSource request幅がGraph・Planの双方へ記録されることを確認する。"""

    clocked = cw.Flow([1]).rate(4)
    graph_path = tmp_path / "rate_graph.json"
    plan_path = tmp_path / "rate_plan.json"
    clocked.export(graph_path)
    cw.compile([clocked]).export(plan_path)

    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert graph_payload["nodes"][-1]["rate_period"] == "1/4"
    assert graph_payload["nodes"][-1]["rate_policy"] == "hold"
    assert plan_payload["times"][-1]["period"] == {"numerator": 1, "denominator": 4}
    assert plan_payload["sources"][0]["request_duration"] == {
        "numerator": 1,
        "denominator": 4,
    }


def test_portable_plan_round_trip_preserves_edges_buffers_and_time() -> None:
    """Plan JSONがPython実体なしで同じportable descriptorへ復元できる。"""

    source = cw.Flow([1, 2])
    left = source.map(lambda value: value + 1)
    right = source.map(lambda value: value * 2)
    joined = left.map(lambda value, *, other: value + other, other=right)
    plan = cw.compile([cw.output(joined, collector=cw.Latest())])

    restored = cw.PortablePlanIR.from_json(plan.portable_ir.to_json())

    assert restored == plan.portable_ir
    assert restored.buffers[0].read_only
    assert restored.buffers[0].max_items == 1
    assert restored.buffers[0].high_watermark == 1
    assert restored.buffers[0].low_watermark == 0
    assert restored.buffers[0].capacity_reasons[0].startswith("producer_burst:")
    assert restored.buffers[0].consumer_cursor_ids == (0, 1)
    assert restored.buffers[0].reclaim_policy == "all_consumers_advanced"
    assert all(binding.abi_version for binding in restored.bindings)


def test_frame_structure_increases_planned_buffer_capacity() -> None:
    """共有祖先だけにframe分岐需要の静的capacityを割り当てる。"""

    source = cw.Flow(range(8))
    framed = source.frame(4)
    merged = source.map(lambda value, *, frame: (value, frame), frame=framed)
    plan = cw.compile([merged])

    capacities = {buffer.producer_port_id: buffer.max_items for buffer in plan.portable_ir.buffers}
    assert capacities == {source.port_id: 4, framed.port_id: 1, merged.port_id: 1}
    source_buffer = plan.portable_ir.buffers[source.port_id]
    assert source_buffer.high_watermark == 4
    assert source_buffer.low_watermark == 3
    assert (
        "shared_merge_demand:node=2:max_items=4:structural_items=4:producer_burst=1"
        in source_buffer.capacity_reasons
    )


def test_emit_many_capacity_is_local_to_producer_port() -> None:
    """Kernel burst上限を無関係な上流Portへ伝播しない。"""

    source = cw.Flow([1])
    expanded = source.map(lambda value: cw.emit_many([value] * 3), max_items=3)
    plan = cw.compile([expanded])

    capacities = {buffer.producer_port_id: buffer.max_items for buffer in plan.portable_ir.buffers}
    assert capacities == {source.port_id: 1, expanded.port_id: 3}


def test_nested_frame_demand_is_multiplied_only_on_shared_ancestor() -> None:
    """nested frameの一件生成需要を共有Sourceまで逆伝播する。"""

    source = cw.Flow(range(12))
    first = source.frame(2)
    nested = first.frame(3)
    merged = source.map(lambda value, *, frame: (value, frame), frame=nested)
    plan = cw.compile([merged])

    capacities = {buffer.producer_port_id: buffer.max_items for buffer in plan.portable_ir.buffers}
    assert capacities == {
        source.port_id: 6,
        first.port_id: 1,
        nested.port_id: 1,
        merged.port_id: 1,
    }
    result = plan.run()
    assert any(item.code == "STALLED_EXACT_MERGE" for item in result.diagnostics)
    assert not any(item.code == "SCHEDULER_DEADLOCK" for item in result.diagnostics)


def test_merge_capacity_rounds_up_to_atomic_producer_burst() -> None:
    """frame需要を満たすcapacityをproducerの原子的burst単位へ切り上げる。"""

    source = cw.Flow([1, 2])
    expanded = source.map(lambda value: cw.emit_many([value] * 3), max_items=3)
    framed = expanded.frame(4)
    merged = expanded.map(lambda value, *, frame: (value, frame), frame=framed)
    plan = cw.compile([merged])

    capacities = {buffer.producer_port_id: buffer.max_items for buffer in plan.portable_ir.buffers}
    assert capacities == {
        source.port_id: 1,
        expanded.port_id: 6,
        framed.port_id: 1,
        merged.port_id: 1,
    }
    result = plan.run()
    assert any(item.code == "STALLED_EXACT_MERGE" for item in result.diagnostics)
    assert not any(item.code == "SCHEDULER_DEADLOCK" for item in result.diagnostics)
