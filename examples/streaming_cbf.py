"""固定shape sample入力を論理時間でframe化し、固定CBFへ流す最小例。"""

from __future__ import annotations

import chronowire as cw
from chronowire_reference import fixed_cbf

Sample = tuple[float, ...]
BeamFrame = tuple[tuple[float, ...], ...]


def run_example() -> tuple[BeamFrame, ...]:
    """2 channelの二つのchunkを4 Hz、4 sample frameの固定CBFで処理する。"""

    values: tuple[Sample, ...] = (
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (4.0, 4.0),
        (5.0, 5.0),
        (6.0, 6.0),
    )
    samples = tuple(
        cw.Emission(
            value,
            cw.LogicalInterval(
                cw.LogicalTime(index, 1, 4),
                cw.LogicalTime(index + 1, 1, 4),
            ),
            index,
        )
        for index, value in enumerate(values)
    )
    frames = cw.Flow(cw.f64_vector_source(samples, width=2)).rate(4).frame(4, pad_end=True)
    beams = fixed_cbf(frames, ((0.5, 0.5),))
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
