# Chronowire 設計書

Chronowire は、論理時間に基づくストリーミング処理を Flow API で記述し、計算グラフとしてコンパイル・実行するための軽量フレームワークである。

Chronowire は spflow の後継や互換レイヤではなく、独立したフレームワークである。Chainer からはスコープ付き設定など一部のAPI設計を参考にするが、Define-by-Runは採用しない。Flow構築時には計算せず、明示的な`compile()`で実行計画を生成する。

Chronowire 自体は DSP アルゴリズム集ではない。FFT、FIR、ビームフォーマ、共分散推定、MVDR などは外部 Kernel パッケージが担当し、Chronowire は以下に責務を限定する。

- Flow による宣言的な処理記述
- 論理時間と時間区間の管理
- グラフ構築
- 分岐・合流・同期
- バッファ管理
- Scheduler
- `chronowire.compile()`
- ExecutionPlan
- 最適化
- Backend 抽象化
- Extension
- `graph_info()`
- `export()`

## 文書一覧

1. [01_基本設計.md](01_基本設計.md)
2. [02_公開API設計.md](02_公開API設計.md)
3. [03_内部グラフと論理時間設計.md](03_内部グラフと論理時間設計.md)
4. [04_compileと実行時設計.md](04_compileと実行時設計.md)
5. [05_Kernel_Backend_Config設計.md](05_Kernel_Backend_Config設計.md)
6. [06_Extension_export_診断設計.md](06_Extension_export_診断設計.md)
7. [07_パッケージ構成と実装計画.md](07_パッケージ構成と実装計画.md)
8. [08_試験設計.md](08_試験設計.md)
9. [09_v0.1確定事項と将来検討.md](09_v0.1確定事項と将来検討.md)
10. [10_Native_Executor設計方針.md](10_Native_Executor設計方針.md)
11. [11_Buffer_Scheduler_PortablePlanIR設計.md](11_Buffer_Scheduler_PortablePlanIR設計.md)
12. [12_v0.1_v0.2リリース方針.md](12_v0.1_v0.2リリース方針.md)

## 最小利用イメージ

```python
import chronowire as cw

config = cw.Config(
    system={"fs": 32768},
)

source = cw.Flow(scene_source, config)
base = source.map(preprocess)

beam = (
    base
    .frame(size=1024, hop=512)
    .map(fft_kernel)
    .map(beamformer_kernel)
)

covariance = (
    base
    .frame(size=2048, hop=1024)
    .map(covariance_kernel)
)

plan = cw.compile(
    [
        cw.output(beam, collector=cw.Latest()),
        cw.output(covariance, collector=cw.Bounded(max_items=64)),
    ],
    extensions=[
        cw.Snapshot(
            flow=covariance,
            path="snapshots/covariance",
            include_degraded=True,
        ),
    ],
)

run_result = plan.run(duration=60.0)
beam_result, covariance_result = run_result.outputs
plan.export("execution_plan.json")
```

`compile()`に指定されたFlowまたはOutputSpecは、そのExecutionPlanにおける観測終端となる。出力値はcollectorを明示した範囲だけ保持する。Extensionは途中Flowも観測でき、正常値だけでなく安全に生成された劣化結果と診断も保存できる。
