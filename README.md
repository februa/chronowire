# Chronowire

Chronowireは、論理時間付きストリームをFlow APIでLogical Graphとして構築し、明示的にcompileして実行するPythonフレームワークである。

公開語彙は`Flow → compile() → Plan → create_session() → Session`、処理部品は
`Operation → Kernel → KernelState`へ統一する。通常のFlow利用者はOperationだけを扱い、
Kernelとfixed-schema/zero-copy宣言はDSP package実装者向けの高度な境界である。
OperationSpec中心APIでは、宣言、Python参照実装、名前付き入力、
Config subtree、compile-time shape検証まで利用できる。PortablePlanIRのOperationDescriptorとnative
ImplementationDescriptorはschema 0.4として実装済みであり、PythonまたはC ABI
ImplementationBindingから再bindできる。固定shape float64 Operation向けのversion付きmodule table、
共有library loader、PythonExecutorからのC ABI conformance実行、CppExecutor動的呼出しも利用できる。
Operation実装の選択はcompile時、Executorの選択はrun時の独立した軸である。詳細は
[14_Operation設計.md](doc/chronowire_design/14_Operation設計.md)を参照する。

## v0.1

- 不変なスコープ付き`Config`
- finite/generated `Source`
- `Flow.map()`、`Flow.frame()`、`Flow.rate()`、同期Flow引数、latest `StateFlow`
- 0/1/複数Emissionと`OK`/`DEGRADED`/`INVALID`伝播
- `Operation`、不変な`Kernel`、run-local `KernelState`、`PythonBackend`
- Python callableの`KernelState`境界への正規化
- required Node抽出、interval不一致warning、単一thread決定的demand-driven runtime
- 読み取り専用の共有`PortBuffer`、consumer cursor、Port別の静的capacity根拠
- Node/Port/Edge/Buffer/Time/Source/Extension/BindingのPortablePlanIR round-trip
- `PORT_SHARED`、`FRAME_HISTORY`、`LATEST_STATE`のrun-local buffer実体
- `RealtimeSource`、bounded `REALTIME_INGRESS`、dropを保持するGapMarker
- realtime欠落後の`DEGRADED`伝播、FRAME履歴破棄、KernelState reset
- `NoCollect`、`Latest`、`Bounded`、`Sink`
- `observe(extension_id=...)`で固定する観測境界とrun-local Extension binding
- 論理時間trigger `EveryLogicalTime`、劣化結果を保存する`Snapshot`
- Logical Graph/PlanのJSON・DOT export

v0.1はPython Executorの基準意味論、gap後のexact merge再同期、golden conformance trace、PortablePlanIR round-tripを固定した。継続sessionやNative Executorへ進んでも、この値、時間、status、Diagnostic、buffer寿命の契約を維持する。

v0.1のmergeは完全interval一致とlatest stateに限定する。不一致の可能性はcompile warningとし、runtimeは必要intervalを生成できない経路を`STALLED_EXACT_MERGE`として停止する。

`rate()`は論理時間上の発火周期を有理数で管理する。v0.1のHOLD policyは数値補間を行わず、発火時点を含む入力intervalの値を出力する。generated Sourceへのrequest幅も、下流で必要な最短rate周期から決定する。

同一EmissionはExtension、観測終端collector、下流consumerの順で配送する。collector overflow時も、Extensionは対象Emissionを先に保存できる。

## v0.2

`0.2.0`では、v0.1の基準意味論を継続sessionと実運用streamingへ拡張した。

```python
session = plan.create_session()
session.start()
partial = session.run_until(10)
continued = session.run_until(20)
final = session.close()
```

`Session.start()`で段階実行を選ぶと、同一Session内でKernelState、FRAME/RATE履歴、buffer、collector、Extension triggerを保持する。`Session.run()`を繰り返す場合は呼出しごとにこれらを作り直す。`run_until()`は境界外のSource Emissionを失わず保留し、結果はsession開始時からの累積snapshotとして返す。`cancel()`はpending状態をdrainせず破棄し、`SESSION_CANCELLED` Diagnosticを残す。

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
Python Stage境界として明示する。CppExecutorは、all-Python Planを単一の最大Python islandとし、
C++側の`advance()` / `resume()`状態とGIL下のadapterを通して実行できる。Python/native mixed
PlanはPython islandの前後にnative区間を置ける。通常のPython実装は固定shape f64 batchを境界で
一回copyするが、`@cw.operation(..., accepts_readonly_buffers=True)`を明示した実装にはC++所有値を
read-only一次元`memoryview`として貸し、同じviewを返す連続batchはnative suffixへcopyせずbindする。
このopt-in実装はExecutor非依存性のため、通常のPython値とmemoryviewの両方を受理する。
Sourceを含むPython prefixまたはnative prefixのどちらから始まる線形Planでも、複数Python islandを
実行できる。native区間からPython islandへ入る複数Portは`synchronous`と`latest`をStage単位で
照合できる。Pythonからnativeへの複数ingress、計算で新規生成したbufferのzero-copy、mixed継続Sessionは
段階実装中である。

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
runtimeで運用できる。version付きKernel ABI tableにはidentity f64、固定CBF、および周期更新MVDRの
参照ABIを登録する。C++実行中は
GILを解放し、NoCollectではKernel出力値をPythonへ戻さない。

```python
import chronowire as cw
from chronowire_reference import CythonCbfBackend, fixed_cbf, fixed_cbf_operation

source = cw.Flow(cw.f64_vector_source([(1.0, 1.0), (2.0, 2.0)], width=2))
beam = fixed_cbf(source.rate(1).frame(2), ((0.5, 0.5),))
plan = cw.compile(
    [cw.output(beam, collector=cw.Latest())],
    implementations={fixed_cbf_operation.operation_id: CythonCbfBackend()},
)

result = plan.run(executor="cpp")
```

周期更新MVDRの受入例は、完成frameから`sample_every(update_period)`で更新境界だけを選択し、累積
共分散、MVDR重み生成、latest StateFlowによる重み保持、各frameへの重み適用を一つのschema 0.4 Planへ
compileする。`sample_every()`は数値rate変換ではなく、frameを分割・複製しないEmission選択である。
更新周期がframe周期の整数倍でなければcompile errorにする。初期積分がchannel数の二乗未満でも処理を
止めず、重みとbeamを`DEGRADED`かつ`INSUFFICIENT_INTEGRATION`付きで保存する。

このMVDRはScheduler、複数入力、LATEST、state resetのconformance用実数参照実装であり、本番用の複素
FFT bin別DSP実装ではない。PythonExecutorとCppExecutorは値、interval、sequence、status、Diagnosticを
同じtraceとして返し、同じPlanの再実行では共分散積分状態を持ち越さない。

外部DSP packageは`cw.native_operation_include_dir()`で配布済みinclude directoryを取得し、
[`native_operation_abi.h`](src/chronowire/native_operation_abi.h)をincludeする。
`chronowire_operation_module_v1`をexportする。共有libraryはprocess-localにloadし、Backendへ渡す。

```python
module = cw.NativeOperationModule("/path/to/libmy_dsp.so")
backend = cw.NativeModuleBackend(module)
plan = cw.compile(
    outputs,
    backend="python",  # 未指定Operationの既定Implementation
    implementations={"my_dsp.beamformer.v1": backend},
)
reference_result = plan.run(executor="python")

# 全OperationがCppExecutor対応ImplementationならPlan運用もnative化できる。
native_plan = cw.compile(outputs, backend=backend)
native_result = native_plan.run(executor="cpp")
```

`CppSession.last_metrics` は、native run中のGIL解放契約、Stage内Python dispatch、
Python境界callback、公開Emission復元数、batch変換数を分離して記録する。
`python_free_hot_path` がTrueでもRunResultやExtension境界のPython処理は隠さず、別counterとして報告する。

別processでschema 0.4 IRを読む場合は、各`implementation:*` slotへ`module.binding(operation_id)`を
指定して`bind_plan(..., backend=backend)`する。library path、module handle、function pointerは
PortablePlanIRへ保存しない。module ABI v1は固定shape、連続float64、単一output、一入力あたり一Emissionを
対象とし、Configは`ConfigSpec.fields`順のfloat64 scalar/tupleとして`create`へ渡す。PythonExecutorは
選択済みC ABI ImplementationをEmission単位で直接呼び、Python計算実装へfallbackしない。この経路は
意味論照合とデバッグ用であり、完全native hot pathにはCppExecutorを選ぶ。

Flow APIとcompileはPython、compile後のlogical tick、Scheduler、buffer、CBF、collector保持選択は
C++が所有する。Kernel定数は`NativeKernelRuntimeBinding`としてprocess-local slotへbindし、
PortablePlanIRへPython objectやpointerを保存しない。共通祖先のbatchはread-onlyとしてfan-out間で
共有し、一回のrunでNodeを一度だけ評価する。`INPUT_OVERRUN`ではRATE cursorとFRAME履歴をresetし、
INVALID、DEGRADED、metadata、Diagnostic provenanceをPython基準意味論と一致させる。

`create_session(executor="cpp")`は有限Sourceを排他的論理時間境界ごとに観測し、flush、close、
cancelのlifecycleを提供する。one-shot実行ではcompile済み観測PortをC++で取得した後、Python Stage境界で
Extensionへpriority順に配送する。

v0.4の完全native経路は単一有限Source、`FRAME(pad_end=False)`、既定`RuntimeOptions`に限定する。
Python Operation／plain callableを含むPlanは協調的なPython islandとしてCppExecutorで実行できるが、
realtime push、複数Source/merge、mixed Planのincremental cursor、継続Sessionで状態を持つExtension、
manifestから読み込む汎用mutable native workspaceは暗黙fallbackせず明示エラーにする。
登録済みMVDR参照ABIのrun-local共分散状態だけは受入経路として対応する。残りはv0.5で
実測しながら拡張する。

## 実行例

固定shape sample入力をrate、frame、EOF padding、固定CBFへ流す例を実行できる。

```bash
uv run python -m examples.streaming_cbf
```

## Install

0.x releaseはGit tagを指定してrevisionを固定できる。

```bash
uv add "chronowire @ git+https://github.com/februa/chronowire.git@v0.4.0"
```

```bash
python -m pip install "chronowire @ git+https://github.com/februa/chronowire.git@v0.4.0"
```

`0.4.0`を含む0.x版はGitHub tagからinstallする。PyPIへは、Python/C++ Executor、継続streaming、評価例、API usabilityを確認したv1.0から正式公開する。release方針は[RELEASING.md](RELEASING.md)を参照する。

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
