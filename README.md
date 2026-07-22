# Chronowire

Chronowireは、論理時間付きストリームをFlow APIでLogical Graphとして構築し、明示的にcompileして実行するPythonフレームワークである。

v0.4の互換実装は`Kernel`/`CompiledKernel`名称を使用する。現在はFlow利用者へABIやSessionを
露出しないOperationSpec中心APIへの段階移行を開始しており、宣言、Python参照実装、名前付き入力、
Config subtree、compile-time shape検証まで利用できる。PortablePlanIRのOperationDescriptorとnative
module bindingは未実装である。詳細は
[14_Operation設計.md](doc/chronowire_design/14_Operation設計.md)を参照する。

## v0.1

- 不変なスコープ付き`Config`
- finite/generated `Source`
- `Flow.map()`、`Flow.frame()`、`Flow.rate()`、同期Flow引数、latest `StateFlow`
- 0/1/複数Emissionと`OK`/`DEGRADED`/`INVALID`伝播
- `Kernel`/`CompiledKernel`、run-local `CompiledKernelSession`、`PythonBackend`
- Python callableの`CompiledKernelSession`境界への正規化
- required Node抽出、interval不一致warning、単一thread決定的demand-driven runtime
- 読み取り専用の共有`PortBuffer`、consumer cursor、Port別の静的capacity根拠
- Node/Port/Edge/Buffer/Time/Source/Extension/BindingのPortablePlanIR round-trip
- `PORT_SHARED`、`FRAME_HISTORY`、`LATEST_STATE`のrun-local buffer実体
- `RealtimeSource`、bounded `REALTIME_INGRESS`、dropを保持するGapMarker
- realtime欠落後の`DEGRADED`伝播、FRAME履歴破棄、Kernel session reset
- `NoCollect`、`Latest`、`Bounded`、`Sink`
- `observe(extension_id=...)`で固定する観測境界とrun-local Extension binding
- 論理時間trigger `EveryLogicalTime`、劣化結果を保存する`Snapshot`
- Logical Graph/ExecutionPlanのJSON・DOT export

v0.1はPython Executorの基準意味論、gap後のexact merge再同期、golden conformance trace、PortablePlanIR round-tripを固定した。継続sessionやNative Executorへ進んでも、この値、時間、status、Diagnostic、buffer寿命の契約を維持する。

v0.1のmergeは完全interval一致とlatest stateに限定する。不一致の可能性はcompile warningとし、runtimeは必要intervalを生成できない経路を`STALLED_EXACT_MERGE`として停止する。

`rate()`は論理時間上の発火周期を有理数で管理する。v0.1のHOLD policyは数値補間を行わず、発火時点を含む入力intervalの値を出力する。generated Sourceへのrequest幅も、下流で必要な最短rate周期から決定する。

同一EmissionはExtension、観測終端collector、下流consumerの順で配送する。collector overflow時も、Extensionは対象Emissionを先に保存できる。

## v0.2

`0.2.0`では、v0.1の基準意味論を継続sessionと実運用streamingへ拡張した。

```python
session = plan.create_plan_session()
session.start()
partial = session.run_until(10)
continued = session.run_until(20)
final = session.close()
```

同一`PlanSession`ではKernel session、FRAME/RATE履歴、buffer、collector、Extension triggerを保持する。別の`PlanSession`は新しいrun-local状態から開始する。`run_until()`は境界外のSource Emissionを失わず保留し、結果はsession開始時からの累積snapshotとして返す。`cancel()`はpending状態をdrainせず破棄し、`SESSION_CANCELLED` Diagnosticを残す。

`close()`はRealtimeSourceを停止してingressを閉じ、残件をdrainする。`cancel()`はdrainせず破棄件数をDiagnosticへ残す。包含、overlap、tolerance同期、複数output Port、StateFlow制御Source、`RuntimeOptions`、PortablePlanIRの明示`ExecutionBindings`、session profilerも公開する。数値resamplingは暗黙に行わず、必要なら外部KernelとしてFlowへ明示する。

RATEとFRAMEは`rate(...).frame(...)`の順で論理格子を確定する。`frame(...).rate(...)`は完成済みframeの重複または未使用を生じ得るためcompile errorとなる。frame列を数値resamplingする場合は、`time_transform="explicit"`のKernelで旧格子を終了し、続く`rate(...).frame(...)`で新しい周期とframe境界を明示する。

```python
left, right = source.map_outputs(split_kernel, output_count=2)
aligned = left.map(
    combine,
    other=right.synchronize(cw.InputSemantics.TOLERANCE, tolerance=0.001),
)
```

PortablePlanIRはprocess-local objectを含めない。別processではslotを完全指定してbindする。

```python
bound = cw.bind_plan(ir, cw.ExecutionBindings(values=slot_bindings, configs=configs))
result = bound.run(options=cw.RuntimeOptions(profiler_enabled=True))
```

## v0.3 development

`0.3.0.dev0`ではNative Executorのsemantic prototypeへ進む。既存runtimeは
`PythonExecutor`として選択可能な境界へ分離し、PortablePlanIR schema 0.3は
`StageDescriptor`、`ValueSchemaDescriptor`、実験的`KernelAbiDescriptor`を持つ。
通常のPython値は`python_opaque`、Kernel ABIは`python-v1`かつnative非互換と明記し、
Native側がdtype、shape、workspaceを推測しない。明示的な`f64_source()`と
`identity_f64()`だけは、限定Cython Executorで実行可能なschemaとABIを宣言する。

```python
result = plan.run(executor=cw.PythonExecutor())
```

最小Cython経路は`SOURCE → RATE(HOLD) → FRAME(pad_end=False) → identity_f64 → collector`
だけを受理する。未対応Node、Python callback、Extension、継続sessionはPythonへ暗黙に
fallbackせず、session作成時に明示エラーにする。

`f64_source()`はscalar列だけでなく、明示interval、status、Diagnosticを持つEmission列も
受理する。Cython Stageは値、論理時刻、sequence、`OK/DEGRADED/INVALID`、Diagnosticを
SoA bufferで運び、Python objectはcollector境界でだけ復元する。`INPUT_OVERRUN`を伴う
gap resetはまだ未対応であり、明示エラーにする。

```python
source = cw.Flow(cw.f64_source([1.0, 2.0, 3.0]))
frames = source.rate(2).frame(2).map(cw.identity_f64())
plan = cw.compile([cw.output(frames, collector=cw.Bounded(8))])

python_result = plan.run(executor=cw.PythonExecutor())
cython_result = plan.run(executor=cw.CythonExecutor())
assert cython_result == python_result
```

sample粒度と4-sample block粒度の固定CBFについて、値、interval、sequence、status、
Diagnosticを同一traceで比較し、Kernel呼出し回数、Scheduler step、Kernel実行時間、
buffer high-watermarkを測定できる。

```bash
uv run python -m examples.native_executor_baseline
```

CBFのPython実装とCython実装は`chronowire_reference`へ本体から分離している。同じKernel宣言を
`PythonBackend`または`CythonCbfBackend`でcompileでき、Python callbackとCython Kernelを
同じPlan内の別Stageとして混在させられる。

```bash
uv run python -m chronowire_reference
```

このconformanceは次の五構成で、値、interval、sequence、status、Diagnosticを比較する。

- Python Executor + Python CBF Kernel
- Python Executor + Cython CBF Kernel
- Python前処理 + Cython CBF Kernel + Python後処理
- Cython Executor + Cython CBF Kernel
- C++ Executor + 固定CBF ABI

Native Executor構成では`f64_vector_source()`が固定channel shapeを宣言する。Cython Executorは
RATE/FRAMEをbatch化してCython CBFへ一回で渡し、C++ ExecutorはRATE/FRAME/CBF/collector保持選択を
一つのC++ sessionで運用する。どちらもEmissionごとのPython dispatchを行わず、CBF出力shapeは
compile時に`beams × frame_size`へ固定される。Python callbackを含む混在Planは引き続き
Python ExecutorがStage境界を管理する。

C++ Executor移行判断用benchmarkは、Plan全体、Source/tick packing、RATE/FRAME、CBF、collector
復元を分離して測定し、copy byteとPython/native境界数をJSONへ保存する。

```bash
uv run python -m benchmarks.native_executor_cpp_gate \
  --sample-count 8192 --block-sizes 64 256 1024 4096 \
  --channels 4 --beams 2 --warmups 5 --repeats 20
```

初回測定とCppExecutorが所有すべき境界は
[13_CppExecutor移行測定.md](doc/chronowire_design/13_CppExecutor移行測定.md)に記録している。

## v0.4 C++ Plan runtime

`0.4.0`では、PythonでcompileしたPortablePlanIRからrun-local C++ sessionを生成し、単一の有限
`f64_vector_source`からなるRATE、FRAME、複数native MAP、fan-out、複数outputを一つのC++ DAG
runtimeで運用できる。version付きKernel ABI tableにはidentity f64と固定CBFを登録する。C++実行中は
GILを解放し、NoCollectではKernel出力値をPythonへ戻さない。

```python
import chronowire as cw
from chronowire_reference import CythonCbfBackend, FixedCbfKernel

source = cw.Flow(cw.f64_vector_source([(1.0, 1.0), (2.0, 2.0)], width=2))
beam = source.rate(1).frame(2).map(FixedCbfKernel(((0.5, 0.5),)))
plan = cw.compile(
    [cw.output(beam, collector=cw.Latest())],
    backend=CythonCbfBackend(),
)

result = plan.run(executor="cpp")
```

Flow APIとcompileはPython、compile後のlogical tick、Scheduler、buffer、CBF、collector保持選択は
C++が所有する。Kernel定数は`NativeKernelRuntimeBinding`としてprocess-local slotへbindし、
PortablePlanIRへPython objectやpointerを保存しない。共通祖先のbatchはread-onlyとしてfan-out間で
共有し、一回のrunでNodeを一度だけ評価する。`INPUT_OVERRUN`ではRATE cursorとFRAME履歴をresetし、
INVALID、DEGRADED、metadata、Diagnostic provenanceをPython基準意味論と一致させる。

`create_plan_session(executor="cpp")`は有限Sourceを排他的論理時間境界ごとに観測し、flush、close、
cancelのlifecycleを提供する。one-shot実行ではcompile済み観測PortをC++で取得した後、Python Stage境界で
Extensionへpriority順に配送する。

v0.4のC++経路は単一有限Source、`FRAME(pad_end=False)`、既定`RuntimeOptions`に限定する。realtime
push、複数Source/merge、任意Python Kernel、native内部のPython callback、継続PlanSessionで状態を持つ
Extension、mutable native Kernel workspaceは暗黙fallbackせず明示エラーにする。これらはv0.5で
実測しながら拡張する。

## 実行例

chunk入力をsampleへ展開し、rate、frame、EOF padding、固定CBFへ流す例を実行できる。

```bash
uv run python -m examples.streaming_cbf
```

## Install

0.x releaseはGit tagを指定してrevisionを固定できる。

```bash
uv add "chronowire @ git+https://github.com/februa/chronowire.git@v0.2.0"
```

```bash
python -m pip install "chronowire @ git+https://github.com/februa/chronowire.git@v0.2.0"
```

`0.1.0`と`0.2.0`はGitHub tagからinstallする。PyPIへは、Python/C++ Executor、継続streaming、評価例、API usabilityを確認したv1.0から正式公開する。release方針は[RELEASING.md](RELEASING.md)を参照する。

## 開発環境

```bash
uv sync --extra dev
uv run pytest
uv run pyright
uv run ruff check .
uv run ruff format --check .
```

Word文書変換skillも使う場合は、docs依存を追加する。

```bash
uv sync --extra dev --extra docs
```

設計の正本は[doc/chronowire_design/README.md](doc/chronowire_design/README.md)を参照する。

v0.1とv0.2のrelease gateは[doc/chronowire_design/12_v0.1_v0.2リリース方針.md](doc/chronowire_design/12_v0.1_v0.2リリース方針.md)に定める。Native Executorの準備実装はv0.3で開始している。
