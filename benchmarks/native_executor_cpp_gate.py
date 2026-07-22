"""C++ Executor移行判断用にCython native経路の境界コストを測定する。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import sysconfig
import time
import tracemalloc
from array import array
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import lcm
from pathlib import Path

import chronowire as cw
from chronowire._cython_executor import run_f64_vector_rate_frame
from chronowire_reference import CythonCbfBackend, fixed_cbf
from chronowire_reference._cython_cbf import run_fixed_cbf_batch

_SCHEMA_VERSION = "0.2"


@dataclass(frozen=True)
class LatencyDistribution:
    """同じ処理のwall-clock latency分布を不変値として保持する。

    Args:
        samples: 記録した反復数。
        minimum_ns: 最小時間。
        p50_ns: nearest-rank 50 percentile。
        p95_ns: nearest-rank 95 percentile。
        p99_ns: nearest-rank 99 percentile。
        maximum_ns: 最大時間。
        mean_ns: 算術平均を整数nsへ丸めた値。

    境界条件:
        warm-upはsamplesへ含めず、全時間を非負のnanosecondで表す。
    """

    samples: int
    minimum_ns: int
    p50_ns: int
    p95_ns: int
    p99_ns: int
    maximum_ns: int
    mean_ns: int


@dataclass(frozen=True)
class CopyAccounting:
    """現Cython経路で静的に特定できるpayload copyと境界を保持する。

    Args:
        source_pack_bytes: Python Sourceからf64 arrayへのcopy量。
        rate_materialization_bytes: RATE出力bufferへのcopy量。
        frame_materialization_bytes: FRAME出力bufferへのcopy量。
        scheduler_return_copy_bytes: Scheduler mallocからPython bytesへのcopy量。
        kernel_return_copy_bytes: Kernel mallocからPython bytesへのcopy量。
        total_payload_copy_bytes: 上記copy量の合計。
        abi_boundary_copy_bytes: owned C ABIで直接除去できるcopy量。
        abi_boundary_copy_percent: 全copyに占めるABI境界copy率。
        payload_copy_count: payload全体をcopyする箇所数。
        python_native_transitions: Python/native境界を跨ぐ回数。
        stage_python_dispatches: native Stage間のPython method dispatch数。
        scheduler_peak_accounted_native_bytes: Schedulerで計数可能な最大byte。
        kernel_peak_accounted_native_bytes: Kernelで計数可能な最大byte。

    境界条件:
        Python object header、allocator metadata、実測RSSは含めない。
    """

    source_pack_bytes: int
    rate_materialization_bytes: int
    frame_materialization_bytes: int
    scheduler_return_copy_bytes: int
    kernel_return_copy_bytes: int
    total_payload_copy_bytes: int
    abi_boundary_copy_bytes: int
    abi_boundary_copy_percent: float
    payload_copy_count: int
    python_native_transitions: int
    stage_python_dispatches: int
    scheduler_peak_accounted_native_bytes: int
    kernel_peak_accounted_native_bytes: int


@dataclass(frozen=True)
class CppExecutorMeasurement:
    """CppExecutorのsession生成込みと構築済みsession実行を分離した測定値。

    Args:
        end_to_end: `Plan.run(executor=CppExecutor())`のlatency。
        session_run: 構築済みrun-local C++ sessionのlatency。
        cython_collected_end_to_end: 全frameをBounded保持するCython latency。
        cpp_collected_end_to_end: 全frameをBounded保持するC++ latency。
        end_to_end_samples_per_second: session生成込みの入力throughput。
        session_samples_per_second: 構築済みsessionの入力throughput。
        speedup_over_cython: Cython/Cpp end-to-end p50比。
        session_speedup_over_cython: Cython/Cpp session p50比。
        collected_speedup_over_cython: Bounded保持時のCython/Cpp p50比。
        runtime_metrics: C++内部区間と境界の直近計測値。
        collected_runtime_metrics: Bounded保持時のC++内部計測値。
        python_heap_peak_bytes: session生成込みrunのPython heap peak。
        collected_python_heap_peak_bytes: Bounded保持時のPython heap peak。

    境界条件:
        NoCollect経路なのでruntime_metricsのoutput boundary byteは0になる。
    """

    end_to_end: LatencyDistribution
    session_run: LatencyDistribution
    cython_collected_end_to_end: LatencyDistribution
    cpp_collected_end_to_end: LatencyDistribution
    end_to_end_samples_per_second: float
    session_samples_per_second: float
    speedup_over_cython: float
    session_speedup_over_cython: float
    collected_speedup_over_cython: float
    runtime_metrics: cw.CppRuntimeMetrics
    collected_runtime_metrics: cw.CppRuntimeMetrics
    python_heap_peak_bytes: int
    collected_python_heap_peak_bytes: int


@dataclass(frozen=True)
class BenchmarkCase:
    """一つのblock sizeについて得たC++移行判断指標を保持する。

    各latencyは独立測定であり、その和をinclusive profileとして扱わない。throughputは
    end-to-end p50から算出し、copy量は固定格子の実装契約から静的に計数する。
    """

    block_size: int
    input_samples: int
    channels: int
    beams: int
    output_frames: int
    end_to_end: LatencyDistribution
    source_materialization_call: LatencyDistribution
    input_time_pack_call: LatencyDistribution
    scheduler_call: LatencyDistribution
    kernel_call: LatencyDistribution
    collector_reconstruction_call: LatencyDistribution
    end_to_end_samples_per_second: float
    end_to_end_frames_per_second: float
    native_calls_p50_ns: int
    native_calls_share_percent: float
    accounted_calls_p50_ns: int
    unattributed_p50_ns: int
    accounted_calls_share_percent: float
    python_heap_peak_bytes: int
    copy_accounting: CopyAccounting
    cpp: CppExecutorMeasurement


@dataclass(frozen=True)
class BenchmarkReport:
    """測定環境、条件、block別結果を持つserialization可能なreport。

    `dataclasses.asdict()`でPython objectを含まないJSON互換dictへ変換できる。性能値は
    environment固有であり、pytestの合否条件または異なるmachine間のgolden値にしない。
    """

    schema_version: str
    benchmark: str
    measured_at_utc: str
    clock: str
    clock_resolution_ns: int
    platform: str
    machine: str
    processor: str
    cpu_model: str
    cpu_count: int | None
    python_version: str
    python_implementation: str
    compiler: str
    compiler_flags: str
    warmups: int
    repeats: int
    cases: tuple[BenchmarkCase, ...]


def _percentile(sorted_values: Sequence[int], percentile: int) -> int:
    index = max(0, (len(sorted_values) * percentile + 99) // 100 - 1)
    return sorted_values[index]


def _measure(
    action: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> LatencyDistribution:
    for _ in range(warmups):
        action()
    gc.collect()
    values: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        action()
        values.append(time.perf_counter_ns() - start)
    ordered = sorted(values)
    return LatencyDistribution(
        repeats,
        ordered[0],
        _percentile(ordered, 50),
        _percentile(ordered, 95),
        _percentile(ordered, 99),
        ordered[-1],
        round(statistics.fmean(ordered)),
    )


def _python_heap_peak(action: Callable[[], object]) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        action()
        _, peak = tracemalloc.get_traced_memory()
        return peak
    finally:
        tracemalloc.stop()


def _values(sample_count: int, channels: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            float((sample_index + channel_index) % 31) / 31.0 for channel_index in range(channels)
        )
        for sample_index in range(sample_count)
    )


def _weights(beams: int, channels: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(beam_index + channel_index + 1) / channels for channel_index in range(channels))
        for beam_index in range(beams)
    )


def _copy_accounting(
    *,
    sample_count: int,
    block_size: int,
    channels: int,
    beams: int,
) -> CopyAccounting:
    frame_count = sample_count // block_size
    source_values = sample_count * channels * 8
    rate_values = source_values
    frame_values = frame_count * block_size * channels * 8
    output_values = frame_count * beams * block_size * 8
    source_metadata = sample_count * (8 + 8 + 1 + 1)
    rate_metadata = sample_count * (8 + 8 + 8 + 1)
    frame_metadata = frame_count * (8 + 8 + block_size * 8 + 1)
    total_copy = source_values + rate_values + frame_values + frame_values + output_values
    abi_boundary_copy = frame_values + output_values
    return CopyAccounting(
        source_values,
        rate_values,
        frame_values,
        frame_values,
        output_values,
        total_copy,
        abi_boundary_copy,
        abi_boundary_copy * 100.0 / total_copy,
        5,
        4,
        1,
        source_values
        + source_metadata
        + rate_values
        + rate_metadata
        + frame_values
        + frame_metadata
        + frame_values,
        frame_values + output_values + output_values,
    )


def _run_case(
    *,
    sample_count: int,
    block_size: int,
    channels: int,
    beams: int,
    warmups: int,
    repeats: int,
) -> BenchmarkCase:
    values = _values(sample_count, channels)
    weights = _weights(beams, channels)
    source_values_contract = cw.f64_vector_source(values, width=channels)
    source = cw.Flow(source_values_contract)
    frames = source.rate(1).frame(block_size)
    output = fixed_cbf(frames, weights)
    plan = cw.compile(
        [cw.output(output, collector=cw.NoCollect())],
        backend=CythonCbfBackend(),
    )
    frame_count = sample_count // block_size
    collected_plan = cw.compile(
        [cw.output(output, collector=cw.Bounded(frame_count))],
        backend=CythonCbfBackend(),
    )
    executor = cw.CythonExecutor()
    cpp_executor = cw.CppExecutor()

    def run_end_to_end() -> object:
        return plan.run(executor=executor)

    first_result = run_end_to_end()
    if not isinstance(first_result, cw.RunResult):
        raise RuntimeError("Cython benchmark did not return RunResult")
    if first_result.outputs[0].received_count != frame_count:
        raise RuntimeError("Cython benchmark frame count does not match the fixed grid")

    def run_cpp_end_to_end() -> object:
        return plan.run(executor=cpp_executor)

    cpp_session = plan.create_session(executor=cpp_executor)
    cpp_collected_session = collected_plan.create_session(executor=cpp_executor)

    def run_cpp_session() -> object:
        return cpp_session.run()

    def run_cython_collected_end_to_end() -> object:
        return collected_plan.run(executor=executor)

    def run_cpp_collected_end_to_end() -> object:
        return collected_plan.run(executor=cpp_executor)

    cpp_reference = run_cpp_session()
    if cpp_reference != first_result:
        raise RuntimeError("CppExecutor benchmark trace differs from CythonExecutor")
    if cpp_collected_session.run() != run_cython_collected_end_to_end():
        raise RuntimeError("CppExecutor collected trace differs from CythonExecutor")

    def run_source_materialization() -> object:
        return source_values_contract.emissions()

    source_emissions = source_values_contract.emissions()

    def run_input_time_pack() -> tuple[
        array[float],
        array[int],
        array[int],
        array[int],
        array[int],
    ]:
        denominator = 1
        for emission in source_emissions:
            denominator = lcm(
                denominator,
                emission.interval.start.as_fraction().denominator,
                emission.interval.end.as_fraction().denominator,
            )
        start_ticks = tuple(
            (emission.interval.start.as_fraction() * denominator).numerator
            for emission in source_emissions
        )
        end_ticks = tuple(
            (emission.interval.end.as_fraction() * denominator).numerator
            for emission in source_emissions
        )
        return (
            array("d", (value for emission in source_emissions for value in emission.value)),
            array("q", start_ticks),
            array("q", end_ticks),
            array("B", [0]) * sample_count,
            array("B", [0]) * sample_count,
        )

    source_values, starts, ends, statuses, resets = run_input_time_pack()

    def run_scheduler() -> object:
        return run_f64_vector_rate_frame(
            memoryview(source_values),
            channels,
            memoryview(starts),
            memoryview(ends),
            memoryview(statuses),
            memoryview(resets),
            1,
            1,
            block_size,
            block_size,
        )

    scheduler_result = run_scheduler()
    if not isinstance(scheduler_result, tuple):
        raise RuntimeError("Cython scheduler benchmark returned an invalid result")
    frame_bytes = scheduler_result[0]
    returned_frame_count = scheduler_result[1]
    if not isinstance(frame_bytes, bytes) or returned_frame_count != frame_count:
        raise RuntimeError("Cython scheduler benchmark returned an invalid frame batch")
    weight_values = array("d", (weight for beam in weights for weight in beam))

    def run_kernel() -> object:
        return run_fixed_cbf_batch(
            memoryview(frame_bytes).cast("d"),
            frame_count,
            block_size,
            channels,
            memoryview(weight_values),
            beams,
        )

    kernel_bytes = run_kernel()
    if not isinstance(kernel_bytes, bytes):
        raise RuntimeError("Cython CBF benchmark returned an invalid output batch")
    output_values = memoryview(kernel_bytes).cast("d")
    frame_starts = scheduler_result[2]
    frame_ends = scheduler_result[3]
    frame_statuses = scheduler_result[4]
    frame_provenance = scheduler_result[5]
    if not all(
        isinstance(item, tuple)
        for item in (frame_starts, frame_ends, frame_statuses, frame_provenance)
    ):
        raise RuntimeError("Cython scheduler benchmark returned invalid metadata")

    def run_collector_reconstruction() -> object:
        collector = cw.NoCollect().create_session()
        output_width = beams * block_size
        for sequence, (start, end, native_status, provenance) in enumerate(
            zip(frame_starts, frame_ends, frame_statuses, frame_provenance, strict=True)
        ):
            if not all(isinstance(item, int) for item in (start, end, native_status)):
                raise RuntimeError("Cython scheduler metadata type changed during benchmark")
            if native_status != 0:
                raise RuntimeError("C++ gate benchmark requires OK-only source status")
            if not isinstance(provenance, tuple):
                raise RuntimeError("Cython scheduler provenance type changed during benchmark")
            value = tuple(
                tuple(
                    float(output_values[sequence * output_width + beam * block_size + index])
                    for index in range(block_size)
                )
                for beam in range(beams)
            )
            diagnostics = tuple(
                diagnostic
                for source_index in provenance
                for diagnostic in source_emissions[source_index].diagnostics
            )
            collector.add(
                cw.Emission(
                    value,
                    cw.LogicalInterval(cw.LogicalTime(start), cw.LogicalTime(end)),
                    sequence,
                    cw.EmissionStatus.OK,
                    diagnostics,
                )
            )
        return collector.snapshot()

    end_to_end = _measure(run_end_to_end, warmups=warmups, repeats=repeats)
    source_materialization_call = _measure(
        run_source_materialization,
        warmups=warmups,
        repeats=repeats,
    )
    input_time_pack_call = _measure(run_input_time_pack, warmups=warmups, repeats=repeats)
    scheduler_call = _measure(run_scheduler, warmups=warmups, repeats=repeats)
    kernel_call = _measure(run_kernel, warmups=warmups, repeats=repeats)
    collector_reconstruction_call = _measure(
        run_collector_reconstruction,
        warmups=warmups,
        repeats=repeats,
    )
    cpp_end_to_end = _measure(run_cpp_end_to_end, warmups=warmups, repeats=repeats)
    cpp_session_run = _measure(run_cpp_session, warmups=warmups, repeats=repeats)
    cython_collected_end_to_end = _measure(
        run_cython_collected_end_to_end,
        warmups=warmups,
        repeats=repeats,
    )
    cpp_collected_end_to_end = _measure(
        run_cpp_collected_end_to_end,
        warmups=warmups,
        repeats=repeats,
    )
    cpp_session.run()
    cpp_metrics = getattr(cpp_session, "last_metrics", None)
    if not isinstance(cpp_metrics, cw.CppRuntimeMetrics):
        raise RuntimeError("CppExecutor benchmark did not expose native runtime metrics")
    cpp_collected_session.run()
    cpp_collected_metrics = getattr(cpp_collected_session, "last_metrics", None)
    if not isinstance(cpp_collected_metrics, cw.CppRuntimeMetrics):
        raise RuntimeError("CppExecutor collected benchmark did not expose native metrics")
    native_calls_p50 = scheduler_call.p50_ns + kernel_call.p50_ns
    accounted_calls_p50 = (
        source_materialization_call.p50_ns
        + input_time_pack_call.p50_ns
        + native_calls_p50
        + collector_reconstruction_call.p50_ns
    )
    unattributed_p50 = max(0, end_to_end.p50_ns - accounted_calls_p50)
    return BenchmarkCase(
        block_size,
        sample_count,
        channels,
        beams,
        frame_count,
        end_to_end,
        source_materialization_call,
        input_time_pack_call,
        scheduler_call,
        kernel_call,
        collector_reconstruction_call,
        sample_count * 1_000_000_000.0 / end_to_end.p50_ns,
        frame_count * 1_000_000_000.0 / end_to_end.p50_ns,
        native_calls_p50,
        native_calls_p50 * 100.0 / end_to_end.p50_ns,
        accounted_calls_p50,
        unattributed_p50,
        accounted_calls_p50 * 100.0 / end_to_end.p50_ns,
        _python_heap_peak(run_end_to_end),
        _copy_accounting(
            sample_count=sample_count,
            block_size=block_size,
            channels=channels,
            beams=beams,
        ),
        CppExecutorMeasurement(
            cpp_end_to_end,
            cpp_session_run,
            cython_collected_end_to_end,
            cpp_collected_end_to_end,
            sample_count * 1_000_000_000.0 / cpp_end_to_end.p50_ns,
            sample_count * 1_000_000_000.0 / cpp_session_run.p50_ns,
            end_to_end.p50_ns / cpp_end_to_end.p50_ns,
            end_to_end.p50_ns / cpp_session_run.p50_ns,
            cython_collected_end_to_end.p50_ns / cpp_collected_end_to_end.p50_ns,
            cpp_metrics,
            cpp_collected_metrics,
            _python_heap_peak(run_cpp_end_to_end),
            _python_heap_peak(run_cpp_collected_end_to_end),
        ),
    )


def run_benchmark(
    *,
    sample_count: int = 8192,
    block_sizes: Sequence[int] = (64, 256, 1024, 4096),
    channels: int = 4,
    beams: int = 2,
    warmups: int = 5,
    repeats: int = 20,
    cpu_model: str | None = None,
) -> BenchmarkReport:
    """Cython経路を測定してC++ Executor移行判断用reportを返す。

    Args:
        sample_count: 一回のrunで処理する入力sample数。
        block_sizes: 比較するFRAME size。sample_countを割り切る必要がある。
        channels: 固定CBFの入力channel数。
        beams: 固定CBFの出力beam数。
        warmups: 記録前のwarm-up回数。
        repeats: latency分布へ記録する反復回数。
        cpu_model: platform APIで識別できない場合に記録するCPU名。

    Returns:
        実行環境、latency分布、throughput、copy量を含むreport。

    Raises:
        ValueError: 件数、shape、block sizeまたは反復回数が不正な場合。
        RuntimeError: native経路が固定格子どおりの結果を返さない場合。

    境界条件:
        block sizeはsample_countを割り切り、EOF paddingや重複FRAMEを測定しない。
    """

    if sample_count <= 0 or channels <= 0 or beams <= 0:
        raise ValueError("sample count, channels, and beams must be positive")
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    normalized_blocks = tuple(block_sizes)
    if not normalized_blocks or any(
        block <= 0 or sample_count % block != 0 for block in normalized_blocks
    ):
        raise ValueError("every block size must be positive and divide sample_count")
    clock = time.get_clock_info("perf_counter")
    cases = tuple(
        _run_case(
            sample_count=sample_count,
            block_size=block_size,
            channels=channels,
            beams=beams,
            warmups=warmups,
            repeats=repeats,
        )
        for block_size in normalized_blocks
    )
    return BenchmarkReport(
        _SCHEMA_VERSION,
        "native_executor_cpp_gate",
        datetime.now(UTC).isoformat(),
        clock.implementation,
        round(clock.resolution * 1_000_000_000),
        platform.platform(),
        platform.machine(),
        platform.processor(),
        cpu_model or platform.processor() or platform.machine(),
        os.cpu_count(),
        platform.python_version(),
        platform.python_implementation(),
        str(sysconfig.get_config_var("CC") or ""),
        str(sysconfig.get_config_var("CFLAGS") or ""),
        warmups,
        repeats,
        cases,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=(64, 256, 1024, 4096))
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--beams", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cpu-model")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI引数で測定し、JSONをstdoutまたは指定fileへ出力する。

    Args:
        argv: 解析する引数。Noneではprocessの引数を使用する。

    Returns:
        正常終了を表す0。

    Raises:
        ValueError: 測定条件が固定格子契約に違反する場合。
        OSError: 指定した出力fileへ書き込めない場合。
        RuntimeError: native実行結果が測定契約と一致しない場合。
    """

    arguments = _parser().parse_args(argv)
    report = run_benchmark(
        sample_count=arguments.sample_count,
        block_sizes=arguments.block_sizes,
        channels=arguments.channels,
        beams=arguments.beams,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
        cpu_model=arguments.cpu_model,
    )
    payload = json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n"
    if arguments.output is None:
        sys.stdout.write(payload)
    else:
        arguments.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
