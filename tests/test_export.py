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

    assert payload["outputs"][0]["collector"] == "Latest"
    assert payload["diagnostics"][0]["code"] == "POSSIBLE_INTERVAL_MISMATCH"
    assert payload["nodes"][-1]["execution_domain"] == "python"
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
    assert plan_payload["nodes"][-1]["rate_period"] == "1/4"
    assert plan_payload["source_request_periods"] == {"0": "1/4"}
