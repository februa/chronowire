"""一定論理間隔で重みを更新する実数MVDR参照Flowを定義する。"""

from __future__ import annotations

from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import chronowire as cw

Matrix = tuple[tuple[float, ...], ...]
Vector = tuple[float, ...]
Frame = tuple[Vector, ...]


def _matrix_shape(
    inputs: Mapping[str, object],
    config: cw.ConfigView,
) -> tuple[int, ...]:
    """frame schemaから正方共分散shapeを解決する。"""

    del config
    schema = inputs["signal"]
    shape = getattr(schema, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("MVDR covariance requires samples x channels")
    return (shape[1], shape[1])


def _weight_shape(
    inputs: Mapping[str, object],
    config: cw.ConfigView,
) -> tuple[int, ...]:
    """正方共分散schemaからweight vector shapeを解決する。"""

    del config
    schema = inputs["covariance"]
    shape = getattr(schema, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("MVDR weights require square covariance")
    return (shape[0],)


def _beam_shape(
    inputs: Mapping[str, object],
    config: cw.ConfigView,
) -> tuple[int, ...]:
    """frame schemaから一beamのsample vector shapeを解決する。"""

    del config
    schema = inputs["signal"]
    shape = getattr(schema, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("MVDR apply requires samples x channels")
    return (shape[0],)


covariance_operation = cw.declare_operation(
    operation_id="chronowire.reference.covariance_accumulator_f64.v1",
    inputs={
        "signal": cw.OperationInputSpec(
            primary=True,
            value=cw.ValueSpec(dtype="float64", shape=("samples", "channels")),
        )
    },
    output=cw.OperationOutputSpec(value=cw.ValueSpec(dtype="float64", shape=(None, None))),
    config=cw.ConfigSpec(scope="mvdr.covariance", fields={"diagonal_loading": float}),
    state="session",
    shape_resolver=_matrix_shape,
)


def _solve(matrix: Matrix, right: Vector) -> Vector:
    """partial pivot付きGauss消去で小規模実数連立方程式を解く。"""

    size = len(right)
    rows = [list(matrix[index]) + [right[index]] for index in range(size)]
    for pivot in range(size):
        selected = max(range(pivot, size), key=lambda row: abs(rows[row][pivot]))
        if abs(rows[selected][pivot]) <= 1.0e-12:
            raise ValueError("MVDR covariance is singular after diagonal loading")
        rows[pivot], rows[selected] = rows[selected], rows[pivot]
        divisor = rows[pivot][pivot]
        for column in range(pivot, size + 1):
            rows[pivot][column] /= divisor
        for row in range(size):
            if row == pivot:
                continue
            factor = rows[row][pivot]
            for column in range(pivot, size + 1):
                rows[row][column] -= factor * rows[pivot][column]
    return tuple(rows[index][size] for index in range(size))


@cw.operation(
    operation_id="chronowire.reference.mvdr_weights_f64.v1",
    inputs={
        "covariance": cw.OperationInputSpec(
            primary=True,
            value=cw.ValueSpec(dtype="float64", shape=("channels", "channels")),
        )
    },
    output=cw.OperationOutputSpec(value=cw.ValueSpec(dtype="float64", shape=(None,))),
    config=cw.ConfigSpec(scope="mvdr.weights", fields={"steering": tuple}),
    shape_resolver=_weight_shape,
)
def mvdr_weights_operation(inputs: Mapping[str, object], config: cw.ConfigView) -> Vector:
    """`R^-1 a / (a^T R^-1 a)`で実数MVDR重みを生成する。"""

    covariance = inputs["covariance"]
    steering = config.steering
    if not isinstance(covariance, tuple) or not isinstance(steering, tuple):
        raise ValueError("MVDR covariance and steering must be tuples")
    matrix = tuple(tuple(float(value) for value in row) for row in covariance)
    vector = tuple(float(value) for value in steering)
    if not matrix or len(matrix) != len(vector) or any(len(row) != len(vector) for row in matrix):
        raise ValueError("MVDR covariance and steering shape mismatch")
    solved = _solve(matrix, vector)
    denominator = sum(left * right for left, right in zip(vector, solved, strict=True))
    if abs(denominator) <= 1.0e-12:
        raise ValueError("MVDR distortionless denominator is zero")
    return tuple(value / denominator for value in solved)


@cw.operation(
    operation_id="chronowire.reference.apply_weights_f64.v1",
    inputs={
        "signal": cw.OperationInputSpec(
            primary=True,
            value=cw.ValueSpec(dtype="float64", shape=("samples", "channels")),
        ),
        "weights": cw.OperationInputSpec(
            mode="latest",
            value=cw.ValueSpec(dtype="float64", shape=("channels",)),
        ),
    },
    output=cw.OperationOutputSpec(value=cw.ValueSpec(dtype="float64", shape=(None,))),
    shape_resolver=_beam_shape,
)
def apply_weights_operation(inputs: Mapping[str, object], config: cw.ConfigView) -> Vector:
    """latest MVDR重みをframeの各sampleへ適用する。"""

    del config
    signal = inputs["signal"]
    weights = inputs["weights"]
    if not isinstance(signal, tuple) or not isinstance(weights, tuple):
        raise ValueError("MVDR signal and weights must be tuples")
    normalized = tuple(tuple(float(value) for value in item) for item in signal)
    vector = tuple(float(value) for value in weights)
    if not vector or any(len(item) != len(vector) for item in normalized):
        raise ValueError("MVDR signal and weight channel count mismatch")
    return tuple(
        sum(value * weight for value, weight in zip(item, vector, strict=True))
        for item in normalized
    )


@dataclass(frozen=True)
class MvdrFlow:
    """MVDR受入試験で観測するFlow群を保持する。

    各fieldは値そのものではなくGraph上のPortを指す。`frames`と`covariance`は
    中間観測用、`weight_updates`は周期更新、`beam`はlatest重み適用後の出力である。
    instanceはimmutableであり、run-localな共分散積分状態を保持しない。
    """

    frames: cw.Flow[Any]
    covariance: cw.Flow[Any]
    weight_updates: cw.Flow[Any]
    beam: cw.Flow[Any]


def build_mvdr_flow(
    samples: Sequence[Vector],
    *,
    frame_size: int = 2,
    update_period: int | float | Fraction = 4,
    diagonal_loading: float = 0.25,
    steering: Vector = (1.0, 1.0),
) -> MvdrFlow:
    """一定論理間隔で重みを更新しlatest適用するMVDR Flowを構築する。

    Args:
        samples: 時系列順の固定channel実数sample。
        frame_size: 共分散とbeamformingのframe sample数。
        update_period: MVDR重み更新の論理時間間隔。
        diagonal_loading: 各frame共分散へ加える正の対角値。
        steering: distortionless constraintの実数steering vector。

    Returns:
        compile対象の中間Flowと最終beam Flow。

    Raises:
        ValueError: 入力shapeまたは設定値が不正な場合。

    境界条件:
        更新間隔はwall-clockでなく`sample_every(update_period)`としてGraphへ記録する。
        完成frameを分割または複製せず、安定した整数境界にあるEmissionだけを更新へ渡す。
    """

    if not samples or not steering or any(len(item) != len(steering) for item in samples):
        raise ValueError("MVDR samples and steering must have the same positive channel count")
    config = cw.Config(
        mvdr={
            "covariance": {"diagonal_loading": float(diagonal_loading)},
            "weights": {"steering": tuple(float(value) for value in steering)},
        }
    )
    source = cw.Flow(cw.f64_vector_source(samples, width=len(steering)), config)
    frames = source.rate(1).frame(frame_size, hop=frame_size)
    covariance = frames.map(covariance_operation)
    weight_updates = covariance.sample_every(update_period).map(mvdr_weights_operation)
    beam = frames.map(apply_weights_operation, weights=weight_updates.latest())
    return MvdrFlow(frames, covariance, weight_updates, beam)


@dataclass(frozen=True)
class _NativeMvdrState:
    """PythonExecutor conformance用に同一Operation関数を呼ぶnative session。"""

    implementation: object
    config: cw.ConfigView
    operation_id: str

    def process(self, inputs: tuple[object, ...], context: cw.RunContext) -> object:
        """物理入力tupleをOperation宣言順のmappingへ変換して実行する。"""

        del context
        if not callable(self.implementation):
            raise TypeError("MVDR native reference implementation must be callable")
        names = {
            "chronowire.reference.mvdr_weights_f64.v1": ("covariance",),
            "chronowire.reference.apply_weights_f64.v1": ("signal", "weights"),
        }[self.operation_id]
        return self.implementation(dict(zip(names, inputs, strict=True)), self.config)


class _CovarianceAccumulatorState:
    """run内の全sample外積を累積し、各frame境界で共分散を出力する。"""

    def __init__(self, diagonal_loading: float) -> None:
        self._diagonal_loading = diagonal_loading
        self._sums: list[float] = []
        self._sample_count = 0
        self._channel_count = 0

    def process(self, inputs: tuple[object, ...], context: cw.RunContext) -> object:
        """現在frameを累積し、不十分な積分もDEGRADED共分散として返す。"""

        signal = inputs[0]
        if not isinstance(signal, tuple) or not signal:
            raise ValueError("covariance signal must be a non-empty frame")
        rows = tuple(tuple(float(value) for value in item) for item in signal)
        channel_count = len(rows[0])
        if channel_count == 0 or any(len(item) != channel_count for item in rows):
            raise ValueError("covariance frame channel count must be fixed")
        if self._channel_count == 0:
            self._channel_count = channel_count
            self._sums = [0.0] * (channel_count * channel_count)
        elif self._channel_count != channel_count:
            raise ValueError("covariance channel count changed within a run")
        for item in rows:
            for row in range(channel_count):
                for column in range(channel_count):
                    self._sums[row * channel_count + column] += item[row] * item[column]
        self._sample_count += len(rows)
        covariance = tuple(
            tuple(
                self._sums[row * channel_count + column] / self._sample_count
                + (self._diagonal_loading if row == column else 0.0)
                for column in range(channel_count)
            )
            for row in range(channel_count)
        )
        if self._sample_count >= channel_count * channel_count:
            return covariance
        diagnostic = cw.Diagnostic(
            cw.Severity.WARNING,
            "INSUFFICIENT_INTEGRATION",
            "MVDR covariance has fewer than channels squared samples",
            interval=context.interval,
        )
        return cw.Emission(
            covariance,
            context.interval,
            0,
            cw.EmissionStatus.DEGRADED,
            (diagnostic,),
        )


@dataclass(frozen=True)
class _NativeMvdrKernel:
    """MVDR OperationのPython conformanceとC++ runtime bindingを保持する。"""

    definition: cw.OperationDefinition
    config: cw.ConfigView
    implementation_spec: cw.ImplementationSpec
    parameters: Vector = ()
    parameter_shape: tuple[int, ...] = ()
    abi_version: str = ""
    process_model: str = ""
    workspace_size_bytes: int = 0
    workspace_alignment_bytes: int = 8
    supports_flush: bool = False
    session_local: bool = True
    native_compatible: bool = True
    output_dtype: str = "float64"

    def create_state(self) -> cw.KernelState[object]:
        """run-localなPython conformance sessionを生成する。"""

        if self.implementation_spec.operation_id == covariance_operation.operation_id:
            loading = self.config.diagonal_loading
            if not isinstance(loading, float) or loading < 0.0:
                raise ValueError("diagonal_loading must be a non-negative float")
            return _CovarianceAccumulatorState(loading)

        binding = self.definition.python_binding
        if binding is None:
            raise RuntimeError("MVDR reference Operation lacks Python implementation")
        return _NativeMvdrState(
            binding.implementation,
            self.config,
            self.implementation_spec.operation_id,
        )

    def create_native_runtime_binding(self) -> cw.NativeKernelRuntimeBinding:
        """C++ runtimeへimmutable float64 parameterを渡す。"""

        values = array("d", self.parameters)
        if values.itemsize != 8:
            raise RuntimeError("MVDR native binding requires 64-bit double")
        return cw.NativeKernelRuntimeBinding(
            self.abi_version,
            self.process_model,
            "float64",
            self.parameter_shape,
            values.tobytes(),
        )


@dataclass(frozen=True)
class MvdrNativeBackend:
    """三つのMVDR参照Operationを明示native ABIへcompileするBackend。

    実数の小規模受入試験専用であり、本番用の複素FFT bin別MVDR実装は提供しない。
    compile結果はrun-local session factoryとC++ runtime bindingを持つ。
    """

    name: str = "cpp_mvdr_reference"

    def compile_kernel(
        self,
        kernel: object,
        context: cw.CompileContext,
    ) -> cw.Kernel[object]:
        """legacy Kernelは対象外として明示拒否する。"""

        del kernel, context
        raise TypeError("MvdrNativeBackend supports declared MVDR Operations only")

    def compile_operation(
        self,
        operation: cw.OperationSpec,
        context: object,
    ) -> cw.Kernel[object]:
        """operation IDとConfigからnative実装metadataと定数を生成する。

        Args:
            operation: compile対象の言語非依存Operation宣言。
            context: Node、Config、resolved schemaを含むcompile context。

        Returns:
            Python conformance sessionとC++ bindingを生成できるcompile済みOperation。

        Raises:
            TypeError: contextまたはConfig値の型が契約と異なる場合。
            MissingImplementationError: 未登録operation IDを指定した場合。

        境界条件:
            Python callableやrun-local積分状態はPortablePlanIRへ保存しない。
        """

        if not isinstance(context, cw.CompileContext):
            raise TypeError("MVDR Backend requires CompileContext")
        definitions = {
            covariance_operation.operation_id: covariance_operation,
            mvdr_weights_operation.operation_id: mvdr_weights_operation,
            apply_weights_operation.operation_id: apply_weights_operation,
        }
        definition = definitions.get(operation.operation_id)
        if definition is None:
            raise cw.MissingImplementationError(
                f"operation={operation.operation_id} backend={self.name}"
            )
        config = context.config.view(operation.config.scope)
        parameters: Vector = ()
        parameter_shape: tuple[int, ...] = ()
        if operation.operation_id == covariance_operation.operation_id:
            loading = config.diagonal_loading
            if not isinstance(loading, float):
                raise TypeError("diagonal_loading must be float")
            parameters, parameter_shape = (loading,), (1,)
            process_model = "covariance_accumulator_f64_frame"
        elif operation.operation_id == mvdr_weights_operation.operation_id:
            steering = config.steering
            if not isinstance(steering, tuple):
                raise TypeError("steering must be tuple")
            parameters = tuple(float(value) for value in steering)
            parameter_shape = (len(parameters),)
            process_model = "mvdr_weights_f64"
        else:
            process_model = "apply_weights_f64_latest"
        abi_version = f"chronowire.reference.{process_model}.v1"
        implementation = cw.ImplementationSpec(
            operation.operation_id,
            f"{operation.operation_id}.cpp_reference",
            self.name,
            abi_version,
            True,
            process_model,
            "scalar",
            (),
            0,
            8,
            False,
            True,
        )
        return _NativeMvdrKernel(
            definition,
            config,
            implementation,
            parameters,
            parameter_shape,
            abi_version,
            process_model,
        )
