"""streaming CBFのend-to-end例を検証する。"""

from examples.streaming_cbf import run_example


def test_streaming_cbf_runs_through_rate_and_run_local_kernel_states() -> None:
    """固定shape source、rate、frame、EOF padding、CBFを一本のPlanで実行する。"""

    expected = (((1.0, 2.0, 3.0, 4.0),), ((5.0, 6.0, 0.0, 0.0),))
    assert run_example() == expected
    assert run_example() == expected
