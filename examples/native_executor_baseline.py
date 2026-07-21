"""sample粒度とblock粒度の固定CBFを同一論理traceで比較する。"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import chronowire as cw

Sample = tuple[float, ...]
ChannelBlock = tuple[tuple[float, ...], ...]
BeamSample = tuple[float, ...]
BeamFrame = tuple[tuple[float, ...], ...]

_SAMPLE_RATE = 4
_WEIGHTS: tuple[tuple[float, ...], ...] = ((0.5, 0.5),)
_CHANNELS: ChannelBlock = (
    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
)


@dataclass(frozen=True)
class BaselineEmission:
    """Executor間で比較する一つの出力Emission trace。"""

    value: BeamFrame
    start: Fraction
    end: Fraction
    sequence: int
    status: cw.EmissionStatus
    diagnostic_codes: tuple[str, ...]
    metadata: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class BaselineMeasurement:
    """意味論traceと分離したPython実行粒度の測定値。"""

    scheduler_steps: int
    kernel_calls: int
    kernel_total_ns: int
    buffer_high_watermark_items: int


@dataclass(frozen=True)
class BaselineResult:
    """CBF基準traceと実行粒度測定をまとめる。"""

    emissions: tuple[BaselineEmission, ...]
    measurement: BaselineMeasurement


def _diagnostic(interval: cw.LogicalInterval) -> cw.Diagnostic:
    return cw.Diagnostic(
        cw.Severity.WARNING,
        "BASELINE_DEGRADED_INPUT",
        "first baseline interval intentionally carries a safe degraded status",
        interval=interval,
    )


def _sample_inputs() -> tuple[cw.Emission[Sample], ...]:
    emissions: list[cw.Emission[Sample]] = []
    for index in range(8):
        interval = cw.LogicalInterval(
            cw.LogicalTime(index, 1, _SAMPLE_RATE),
            cw.LogicalTime(index + 1, 1, _SAMPLE_RATE),
        )
        emissions.append(
            cw.Emission(
                tuple(channel[index] for channel in _CHANNELS),
                interval,
                index,
                cw.EmissionStatus.DEGRADED if index == 0 else cw.EmissionStatus.OK,
                (_diagnostic(interval),) if index == 0 else (),
            )
        )
    return tuple(emissions)


def _block_inputs() -> tuple[cw.Emission[ChannelBlock], ...]:
    emissions: list[cw.Emission[ChannelBlock]] = []
    for block_index in range(2):
        start = block_index
        interval = cw.LogicalInterval(cw.LogicalTime(start), cw.LogicalTime(start + 1))
        begin = block_index * _SAMPLE_RATE
        end = begin + _SAMPLE_RATE
        emissions.append(
            cw.Emission(
                tuple(channel[begin:end] for channel in _CHANNELS),
                interval,
                block_index,
                (cw.EmissionStatus.DEGRADED if block_index == 0 else cw.EmissionStatus.OK),
                (_diagnostic(interval),) if block_index == 0 else (),
            )
        )
    return tuple(emissions)


def _beam_sample(sample: Sample) -> BeamSample:
    return tuple(
        sum(weight * value for weight, value in zip(beam, sample, strict=True)) for beam in _WEIGHTS
    )


def _pack_sample_frame(frame: tuple[BeamSample | None, ...]) -> BeamFrame:
    if any(item is None for item in frame):
        raise ValueError("baseline sample frames must be complete")
    return tuple(
        tuple(item[beam_index] for item in frame if item is not None)
        for beam_index in range(len(_WEIGHTS))
    )


def _beam_block(block: ChannelBlock) -> BeamFrame:
    sample_count = len(block[0])
    if any(len(channel) != sample_count for channel in block):
        raise ValueError("baseline block channels must have equal length")
    return tuple(
        tuple(
            sum(
                weight * block[channel_index][sample_index]
                for channel_index, weight in enumerate(beam)
            )
            for sample_index in range(sample_count)
        )
        for beam in _WEIGHTS
    )


def _result(run_result: cw.RunResult) -> BaselineResult:
    profile = run_result.profile
    if profile is None:
        raise RuntimeError("native baseline requires profiler output")
    emissions = tuple(
        BaselineEmission(
            emission.value,
            emission.interval.start.as_fraction(),
            emission.interval.end.as_fraction(),
            emission.sequence,
            emission.status,
            tuple(item.code for item in emission.diagnostics),
            tuple(sorted(emission.metadata.items())),
        )
        for emission in run_result.outputs[0].emissions
    )
    return BaselineResult(
        emissions,
        BaselineMeasurement(
            profile.scheduler_steps,
            sum(item.call_count for item in profile.kernels),
            sum(item.total_ns for item in profile.kernels),
            sum(item.high_watermark for item in profile.buffers),
        ),
    )


def run_sample_baseline() -> BaselineResult:
    """sampleごとのCBF呼出しとFRAMEによる基準traceを返す。"""

    samples = cw.Flow(_sample_inputs())
    frames = samples.map(_beam_sample).frame(_SAMPLE_RATE).map(_pack_sample_frame)
    plan = cw.compile([cw.output(frames, collector=cw.Bounded(2))])
    return _result(plan.run(options=cw.RuntimeOptions(profiler_enabled=True)))


def run_block_baseline() -> BaselineResult:
    """4-sample blockごとのCBF呼出しによる基準traceを返す。"""

    blocks = cw.Flow(_block_inputs()).map(_beam_block)
    plan = cw.compile([cw.output(blocks, collector=cw.Bounded(2))])
    return _result(plan.run(options=cw.RuntimeOptions(profiler_enabled=True)))


def main() -> None:
    """両粒度の意味論一致とPython呼出し回数を表示する。"""

    sample = run_sample_baseline()
    block = run_block_baseline()
    if sample.emissions != block.emissions:
        raise RuntimeError("sample and block baseline traces differ")
    print("sample kernel calls:", sample.measurement.kernel_calls)
    print("block kernel calls:", block.measurement.kernel_calls)
    print("sample kernel total ns:", sample.measurement.kernel_total_ns)
    print("block kernel total ns:", block.measurement.kernel_total_ns)


if __name__ == "__main__":
    main()
