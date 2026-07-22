"""Python/Cython Backend交換を検証する固定CBF参照Operation。"""

from __future__ import annotations

from array import array
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import TypeVar

import chronowire as cw
from chronowire.kernel import KernelProvider

from ._cython_cbf import run_fixed_cbf, run_fixed_cbf_batch

Sample = tuple[float, ...]
BeamFrame = tuple[tuple[float, ...], ...]
FrameValue = TypeVar("FrameValue")


def _normalize_weights(
    weights: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """固定CBF係数を有限なbeam-major matrixへ正規化する。"""

    if not weights or not weights[0]:
        raise ValueError("CBF weights must contain at least one beam and channel")
    channel_count = len(weights[0])
    normalized: list[tuple[float, ...]] = []
    for beam in weights:
        if len(beam) != channel_count:
            raise ValueError("all CBF beams must have the same channel count")
        values = tuple(float(value) for value in beam)
        if not all(isfinite(value) for value in values):
            raise ValueError("CBF weights must be finite")
        normalized.append(values)
    return tuple(normalized)


def _normalize_frame(value: object, channel_count: int) -> tuple[Sample, ...]:
    """Python/Cython実装が共有するframe入力契約を検証する。"""

    if not isinstance(value, tuple):
        raise TypeError("CBF input must be a tuple frame")
    samples: list[Sample] = []
    for sample in value:
        if sample is None:
            samples.append((0.0,) * channel_count)
            continue
        if not isinstance(sample, tuple) or len(sample) != channel_count:
            raise ValueError("CBF sample shape must match the weight channel count")
        normalized = tuple(float(item) for item in sample)
        if not all(isfinite(item) for item in normalized):
            raise ValueError("CBF samples must be finite")
        samples.append(normalized)
    return tuple(samples)


def _cbf_shape(inputs: Mapping[str, object], config: cw.ConfigView) -> tuple[int, ...]:
    """frame schemaと固定係数から`beams x samples` shapeを解決する。"""

    schema = inputs["signal"]
    shape = getattr(schema, "shape", None)
    weights = config.weights
    if not isinstance(shape, tuple) or len(shape) != 2 or not isinstance(weights, tuple):
        raise ValueError("fixed CBF requires samples x channels and tuple weights")
    normalized = _normalize_weights(weights)
    if shape[1] != len(normalized[0]):
        raise ValueError("fixed CBF input channels must match Config weights")
    return (len(normalized), shape[0])


@cw.operation(
    operation_id="chronowire.reference.fixed_cbf_f64.v1",
    inputs={
        "signal": cw.OperationInputSpec(
            primary=True,
            value=cw.ValueSpec(dtype="float64", shape=("samples", "channels")),
        )
    },
    output=cw.OperationOutputSpec(value=cw.ValueSpec(dtype="float64", shape=(None, None))),
    config=cw.ConfigSpec(scope="cbf", fields={"weights": tuple}),
    shape_resolver=_cbf_shape,
)
def fixed_cbf_operation(inputs: Mapping[str, object], config: cw.ConfigView) -> BeamFrame:
    """Configのbeam-major固定係数をsample-major frameへ適用する。"""

    weights = config.weights
    if not isinstance(weights, tuple):
        raise ValueError("fixed CBF weights must be a tuple")
    normalized = _normalize_weights(weights)
    samples = _normalize_frame(inputs["signal"], len(normalized[0]))
    return tuple(
        tuple(
            sum(weight * value for weight, value in zip(beam, sample, strict=True))
            for sample in samples
        )
        for beam in normalized
    )


def fixed_cbf(
    frames: cw.Flow[FrameValue],
    weights: tuple[tuple[float, ...], ...],
) -> cw.Flow[BeamFrame]:
    """固定係数をConfig scopeへ記録してCBF OperationをFlowへ追加する。

    Args:
        frames: sample-major固定shape frame Flow。
        weights: beam-majorの有限float64係数。

    Returns:
        同じGraph上で`fixed_cbf_operation`を適用したbeam Flow。

    Raises:
        ValueError: 係数が空、非矩形、または非有限の場合。

    境界条件:
        係数はprocess-local定数ではなく不変ConfigとしてPlanへ記録し、時変重みには使わない。
    """

    normalized = _normalize_weights(weights)
    configured = frames.with_config(frames.config.scope(cbf={"weights": normalized}))
    return configured.map(fixed_cbf_operation)


@dataclass(frozen=True)
class _CythonCbfState:
    weights: tuple[tuple[float, ...], ...]

    def process(self, inputs: tuple[object, ...], context: cw.RunContext) -> BeamFrame:
        """検証後の固定shape bufferをCython `nogil` loopへ渡す。"""

        del context
        channel_count = len(self.weights[0])
        samples = _normalize_frame(inputs[0], channel_count)
        sample_buffer = array("d", (item for sample in samples for item in sample))
        weight_buffer = array("d", (item for beam in self.weights for item in beam))
        return run_fixed_cbf(
            memoryview(sample_buffer),
            len(samples),
            channel_count,
            memoryview(weight_buffer),
            len(self.weights),
        )

    def process_batch(
        self,
        values: memoryview[float],
        *,
        item_count: int,
        item_shape: tuple[int, ...],
    ) -> cw.NativeValueBatch:
        """複数frameを一回のCython呼出しでCBF変換する。"""

        if len(item_shape) != 2:
            raise ValueError("fixed CBF batch requires frame_size x channel_count shape")
        sample_count, channel_count = item_shape
        if channel_count != len(self.weights[0]):
            raise ValueError("fixed CBF batch channel count does not match weights")
        weight_buffer = array("d", (item for beam in self.weights for item in beam))
        output = run_fixed_cbf_batch(
            values,
            item_count,
            sample_count,
            channel_count,
            memoryview(weight_buffer),
            len(self.weights),
        )
        return cw.NativeValueBatch(
            output,
            item_count,
            (len(self.weights), sample_count),
        )


@dataclass(frozen=True)
class _CythonCbfKernel:
    weights: tuple[tuple[float, ...], ...]
    implementation_spec: cw.ImplementationSpec
    abi_version: str = "chronowire.reference.fixed_cbf_f64.v1"
    process_model: str = "fixed_cbf_f64_frame"
    workspace_size_bytes: int = 0
    workspace_alignment_bytes: int = 8
    supports_flush: bool = False
    session_local: bool = True
    native_compatible: bool = True
    output_dtype: str = "float64"

    def create_state(self) -> _CythonCbfState:
        """run-local Cython CBF sessionを生成する。"""

        return _CythonCbfState(self.weights)

    def create_native_runtime_binding(self) -> cw.NativeKernelRuntimeBinding:
        """CppExecutorへ固定CBF係数をimmutable f64 bindingとして渡す。"""

        weight_buffer = array("d", (item for beam in self.weights for item in beam))
        if weight_buffer.itemsize != 8:
            raise RuntimeError("fixed CBF native binding requires 64-bit double")
        return cw.NativeKernelRuntimeBinding(
            self.abi_version,
            self.process_model,
            "float64",
            (len(self.weights), len(self.weights[0])),
            weight_buffer.tobytes(),
        )

    def resolve_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """`frame_size x channels`を`beams x frame_size`へ解決する。"""

        if len(input_shape) != 2 or input_shape[1] != len(self.weights[0]):
            raise ValueError("fixed CBF input schema must be frame_size x channels")
        return (len(self.weights), input_shape[0])


@dataclass(frozen=True)
class CythonCbfBackend:
    """固定CBF OperationをCython実装へcompileする参照Backend。"""

    name: str = "cython_cbf"

    def compile_kernel(
        self,
        kernel: KernelProvider[object],
        context: cw.CompileContext,
    ) -> cw.Kernel[object]:
        """Operation以外の既存Kernelは標準Python compileへ委譲する。

        Raises:
            Exception: Kernel自身のcompileが失敗した場合。
        """

        return cw.PythonBackend().compile_kernel(kernel, context)

    def compile_operation(
        self,
        operation: cw.OperationSpec,
        context: object,
    ) -> cw.Kernel[object]:
        """Config係数とresolved schemaからCython CBF Kernelを生成する。"""

        if operation.operation_id != fixed_cbf_operation.operation_id:
            raise cw.MissingImplementationError(
                f"operation={operation.operation_id} backend={self.name} "
                "contract=missing_implementation"
            )
        if not isinstance(context, cw.CompileContext):
            raise TypeError("CythonCbfBackend requires CompileContext")
        weights = context.config.view(operation.config.scope).weights
        if not isinstance(weights, tuple):
            raise TypeError("fixed CBF Config weights must be a tuple")
        normalized = _normalize_weights(weights)
        return _CythonCbfKernel(
            normalized,
            cw.ImplementationSpec(
                operation.operation_id,
                "chronowire.reference.fixed_cbf_f64.cython.v1",
                self.name,
                "chronowire.reference.fixed_cbf_f64.v1",
                native_compatible=True,
                process_model="fixed_cbf_f64_frame",
                workspace_size_bytes=0,
                workspace_alignment_bytes=8,
            ),
        )


@dataclass(frozen=True)
class CbfTraceItem:
    """実装間で比較するCBF出力の意味論trace。"""

    value: BeamFrame
    start: Fraction
    end: Fraction
    sequence: int
    status: cw.EmissionStatus
    diagnostic_codes: tuple[str, ...]


@dataclass(frozen=True)
class CbfRun:
    """一つのBackend構成で得たCBF traceとStage配置。"""

    name: str
    trace: tuple[CbfTraceItem, ...]
    stage_domains: tuple[str, ...]
    kernel_abi: str
    native_buffer_count: int
    opaque_port_count: int


def _source() -> tuple[cw.Emission[Sample], ...]:
    """最初のsampleだけを安全なDEGRADEDとして残す2-channel入力を返す。"""

    result: list[cw.Emission[Sample]] = []
    for index in range(8):
        interval = cw.LogicalInterval(
            cw.LogicalTime(index, 1, 4),
            cw.LogicalTime(index + 1, 1, 4),
        )
        diagnostic = cw.Diagnostic(
            cw.Severity.WARNING,
            "CBF_REFERENCE_DEGRADED_INPUT",
            "first reference sample intentionally uses a safe degraded value",
            interval=interval,
        )
        result.append(
            cw.Emission(
                (float(index + 1), float(index + 1)),
                interval,
                index,
                cw.EmissionStatus.DEGRADED if index == 0 else cw.EmissionStatus.OK,
                (diagnostic,) if index == 0 else (),
            )
        )
    return tuple(result)


def _identity_beams(beams: object) -> object:
    """混在Stageを明示するPython後処理境界。"""

    return beams


@cw.operation(
    operation_id="chronowire.reference.identity_sample_f64.v1",
    inputs={
        "value": cw.OperationInputSpec(
            primary=True,
            value=cw.ValueSpec(dtype="float64", shape=("channels",)),
        )
    },
    output="same",
)
def _identity_sample_operation(
    inputs: Mapping[str, object],
    config: cw.ConfigView,
) -> object:
    """固定schemaを保ったままPython Stage境界を作る。"""

    del config
    return inputs["value"]


def _run(
    name: str,
    *,
    backend: str | cw.Backend,
    mixed: bool,
) -> CbfRun:
    source = cw.Flow(
        cw.f64_vector_source(_source(), width=2),
        cw.Config(cbf={"weights": ((0.5, 0.5),)}),
    )
    prepared = source.map(_identity_sample_operation) if mixed else source
    frames = prepared.frame(4)
    beams = frames.map(fixed_cbf_operation)
    result_flow = beams.map(_identity_beams) if mixed else beams
    implementations = {fixed_cbf_operation.operation_id: backend} if backend != "python" else None
    plan = cw.compile(
        [cw.output(result_flow, collector=cw.Bounded(2))],
        implementations=implementations,
    )
    result = plan.run(executor=cw.PythonExecutor())
    map_abis = tuple(item for item in plan.portable_ir.kernel_abis if item.native_compatible)
    kernel_abi = map_abis[0].abi_version if map_abis else "python-v1"
    return CbfRun(
        name,
        tuple(
            CbfTraceItem(
                item.value,
                item.interval.start.as_fraction(),
                item.interval.end.as_fraction(),
                item.sequence,
                item.status,
                tuple(diagnostic.code for diagnostic in item.diagnostics),
            )
            for item in result.outputs[0].emissions
        ),
        tuple(stage.execution_domain for stage in plan.portable_ir.stages),
        kernel_abi,
        len(plan.portable_ir.native_buffers),
        sum(item.value_schema_id == "python:opaque" for item in plan.portable_ir.ports),
    )


def _run_native_executor(name: str, executor: cw.Executor) -> CbfRun:
    """固定shape SourceからCBFまでを指定したnative Executorでbatch実行する。"""

    source = cw.Flow(
        cw.f64_vector_source(_source(), width=2),
        cw.Config(cbf={"weights": ((0.5, 0.5),)}),
    )
    frames = source.rate(4).frame(4)
    beams = frames.map(fixed_cbf_operation)
    plan = cw.compile(
        [cw.output(beams, collector=cw.Bounded(2))],
        implementations={fixed_cbf_operation.operation_id: CythonCbfBackend()},
    )
    result = plan.run(executor=executor)
    abi = next(item for item in plan.portable_ir.kernel_abis if item.native_compatible)
    return CbfRun(
        name,
        tuple(
            CbfTraceItem(
                item.value,
                item.interval.start.as_fraction(),
                item.interval.end.as_fraction(),
                item.sequence,
                item.status,
                tuple(diagnostic.code for diagnostic in item.diagnostics),
            )
            for item in result.outputs[0].emissions
        ),
        tuple(stage.execution_domain for stage in plan.portable_ir.stages),
        abi.abi_version,
        len(plan.portable_ir.native_buffers),
        sum(item.value_schema_id == "python:opaque" for item in plan.portable_ir.ports),
    )


def run_cbf_conformance() -> tuple[CbfRun, CbfRun, CbfRun, CbfRun, CbfRun]:
    """Python/Cython KernelとCython/C++ Executorの五構成を実行する。"""

    return (
        _run("python_cbf", backend="python", mixed=False),
        _run("cython_cbf", backend=CythonCbfBackend(), mixed=False),
        _run("mixed_python_cython", backend=CythonCbfBackend(), mixed=True),
        _run_native_executor("cython_executor_cbf", cw.CythonExecutor()),
        _run_native_executor("cpp_executor_cbf", cw.CppExecutor()),
    )
