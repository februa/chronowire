"""参照Kernel conformanceをcommand lineから実行する。"""

from .cbf import run_cbf_conformance


def main() -> None:
    """Python、Cython、混在CBFの同値性とStage配置を表示する。"""

    runs = run_cbf_conformance()
    reference = runs[0].trace
    if any(item.trace != reference for item in runs[1:]):
        raise RuntimeError("Python/Cython CBF traces differ")
    for item in runs:
        print(item.name, item.kernel_abi, item.stage_domains)


if __name__ == "__main__":
    main()
