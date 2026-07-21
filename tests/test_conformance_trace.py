"""Python Executorのv0.1意味論を固定golden traceと比較する。"""

import json
from fractions import Fraction
from pathlib import Path

import chronowire as cw


def _rational(value: cw.LogicalTime) -> list[int]:
    fraction = value.as_fraction()
    return [fraction.numerator, fraction.denominator]


def _emission_trace(emission: cw.Emission[object]) -> dict[str, object]:
    value = list(emission.value) if isinstance(emission.value, tuple) else emission.value
    return {
        "value": value,
        "interval": [_rational(emission.interval.start), _rational(emission.interval.end)],
        "sequence": emission.sequence,
        "status": emission.status.value,
        "diagnostic_codes": [item.code for item in emission.diagnostics],
        "metadata": emission.metadata,
    }


def test_python_executor_matches_v0_1_golden_trace() -> None:
    """値、時間、sequence、status、Diagnostic、metadataを一括固定する。"""

    diagnostic = cw.Diagnostic(
        cw.Severity.WARNING,
        "SAFE_FALLBACK",
        "insufficient integration used a safe fallback",
        interval=cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1)),
        details={"observed_count": 1},
    )
    source = cw.Flow(
        [
            cw.Emission(
                "safe-fallback",
                cw.LogicalInterval(cw.LogicalTime(0), cw.LogicalTime(1)),
                17,
                cw.EmissionStatus.DEGRADED,
                (diagnostic,),
                {"origin": "conformance", "integration_count": 1},
            )
        ]
    )
    clocked = source.rate(Fraction(2))
    framed = clocked.frame(2)
    result = cw.compile(
        [
            cw.output(clocked, collector=cw.Bounded(2)),
            cw.output(framed, collector=cw.Bounded(1)),
        ]
    ).run()

    trace = {
        "outputs": [
            {
                "collector_kind": output.collector_kind,
                "received_count": output.received_count,
                "dropped_count": output.dropped_count,
                "emissions": [_emission_trace(item) for item in output.emissions],
            }
            for output in result.outputs
        ],
        "status_counts": {
            status.value: count
            for status, count in sorted(
                result.status_counts.items(), key=lambda item: item[0].value
            )
        },
        "diagnostic_codes": [item.code for item in result.diagnostics],
        "completed": result.completed,
    }
    path = Path(__file__).parent / "conformance" / "v0_1_python_trace.json"
    expected = json.loads(path.read_text(encoding="utf-8"))

    assert trace == expected
