"""chunk入力を論理時間でframe化し、固定CBFへ流す最小例。"""

from __future__ import annotations

from dataclasses import dataclass

import chronowire as cw

Sample = tuple[float, ...]
Chunk = tuple[tuple[float, ...], ...]
BeamFrame = tuple[tuple[float, ...], ...]


class _ChunkSamplesSession:
    """一回のrun内でchunkを論理時間付きsampleへ展開するsession。"""

    def __init__(self, sample_rate_hz: int) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._next_sample = 0

    def run(self, inputs: tuple[object, ...], context: cw.RunContext) -> object:
        """channel-first chunkを0件以上のsample Emissionへ変換する。"""

        chunk = inputs[0]
        if not isinstance(chunk, tuple) or not chunk:
            raise ValueError("chunk must be a non-empty channel-first tuple")
        channels = tuple(tuple(float(value) for value in channel) for channel in chunk)
        sample_count = len(channels[0])
        if any(len(channel) != sample_count for channel in channels):
            raise ValueError("all chunk channels must have the same sample count")

        emissions: list[cw.Emission[Sample]] = []
        for offset in range(sample_count):
            index = self._next_sample + offset
            interval = cw.LogicalInterval(
                cw.LogicalTime(index, 1, self._sample_rate_hz),
                cw.LogicalTime(index + 1, 1, self._sample_rate_hz),
            )
            emissions.append(
                cw.Emission(
                    tuple(channel[offset] for channel in channels),
                    interval,
                    index,
                )
            )
        self._next_sample += sample_count
        return cw.emit_many(emissions)


@dataclass(frozen=True)
class _CompiledChunkSamples:
    """sample rateを解決済みのrun-local session factory。"""

    sample_rate_hz: int

    def create_session(self) -> _ChunkSamplesSession:
        """sample cursorが0のsessionを生成する。"""

        return _ChunkSamplesSession(self.sample_rate_hz)


class ChunkSamplesKernel:
    """Configのsample rateでchunkをsample列へ変換するKernel。"""

    def compile(self, context: cw.CompileContext) -> cw.CompiledKernel[object]:
        """`stream.sample_rate_hz`を検証し、session factoryを返す。"""

        sample_rate_hz = context.config.require("stream.sample_rate_hz")
        if not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            raise ValueError("stream.sample_rate_hz must be a positive integer")
        return _CompiledChunkSamples(sample_rate_hz)


class _FixedCbfSession:
    """固定係数をframeごとに適用するrun-local CBF session。"""

    def __init__(self, weights: tuple[tuple[float, ...], ...]) -> None:
        self._weights = weights

    def run(self, inputs: tuple[object, ...], context: cw.RunContext) -> BeamFrame:
        """sample-major frameへ`Y[b,t]=sum_c W[b,c]X[t,c]`を適用する。"""

        frame = inputs[0]
        if not isinstance(frame, tuple):
            raise TypeError("CBF input must be a frame tuple")
        channel_count = len(self._weights[0])
        beams: list[list[float]] = [[] for _ in self._weights]
        for item in frame:
            sample = (0.0,) * channel_count if item is None else item
            if not isinstance(sample, tuple) or len(sample) != channel_count:
                raise ValueError("sample channel count must agree with CBF weights")
            for beam_index, beam_weights in enumerate(self._weights):
                beams[beam_index].append(
                    sum(
                        weight * float(value)
                        for weight, value in zip(beam_weights, sample, strict=True)
                    )
                )
        return tuple(tuple(beam) for beam in beams)


@dataclass(frozen=True)
class _CompiledFixedCbf:
    """検証済み固定CBF係数を保持するsession factory。"""

    weights: tuple[tuple[float, ...], ...]

    def create_session(self) -> _FixedCbfSession:
        """固定係数を共有し、可変状態を持たないsessionを生成する。"""

        return _FixedCbfSession(self.weights)


@dataclass(frozen=True)
class FixedCbfKernel:
    """beam-major固定係数をcompileするCBF Kernel。"""

    weights: tuple[tuple[float, ...], ...]

    def compile(self, context: cw.CompileContext) -> cw.CompiledKernel[object]:
        """係数shapeを検証してCBF session factoryを返す。"""

        if not self.weights or not self.weights[0]:
            raise ValueError("CBF weights must contain beams and channels")
        channel_count = len(self.weights[0])
        if any(len(beam) != channel_count for beam in self.weights):
            raise ValueError("all CBF beams must have the same channel count")
        return _CompiledFixedCbf(self.weights)


def run_example() -> tuple[BeamFrame, ...]:
    """2 channelの二つのchunkを4 Hz、4 sample frameの固定CBFで処理する。"""

    config = cw.Config(stream={"sample_rate_hz": 4})
    chunks: tuple[Chunk, ...] = (
        ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
        ((4.0, 5.0, 6.0), (4.0, 5.0, 6.0)),
    )
    samples = cw.Flow(chunks, config).map(
        ChunkSamplesKernel(),
        config_paths=("stream.sample_rate_hz",),
        max_items=3,
    )
    frames = samples.rate(4).frame(4, pad_end=True)
    beams = frames.map(FixedCbfKernel(weights=((0.5, 0.5),)))
    plan = cw.compile([cw.output(beams, collector=cw.Bounded(2))])
    result = plan.run()
    return tuple(emission.value for emission in result.outputs[0].emissions)


def main() -> None:
    """固定CBFの完成frameとEOF padding frameを表示する。"""

    outputs = run_example()
    if outputs != (((1.0, 2.0, 3.0, 4.0),), ((5.0, 6.0, 0.0, 0.0),)):
        raise RuntimeError(f"unexpected CBF outputs: {outputs!r}")
    print("completed:", outputs[0])
    print("flushed:", outputs[1])


if __name__ == "__main__":
    main()
