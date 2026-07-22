# compileと実行時設計

## 1. compileの責務

`chronowire.compile(outputs)`はLogical GraphからPlanを生成する。

処理段階:

1. 引数検証
2. 出力Port確定
3. 祖先Node抽出
4. 部分Graph生成
5. Graph整合性検証
6. 型・時間意味論検証
7. OperationSpecの入力、shape、Config、status規則検証
8. BackendによるImplementation選択とOperation compile
9. buffer requirement解析
10. Fusion候補解析
11. Batch候補解析
12. parallel stage構築
13. buffer割当
14. output collector設定
15. Plan生成

OperationSpecとImplementationの分離、shape unification、Backend選択は
[14_Operation設計.md](14_Operation設計.md)を正本とする。Backend実装がshapeや時間意味論を独自に
推測してはならない。

## 2. 観測終端

compile指定Portは外部から観測可能な境界になる。値を保持するかどうかはOutputSpecのcollector policyで決まり、観測境界であること自体は全件保持を意味しない。

```python
plan = cw.compile([
    base,
    beam,
])
```

- `base`までを実行
- `base`結果を指定collectorへ渡す
- `beam`までの後段も実行
- 両方のOutputResultを指定順に返す

観測境界ではEmissionをcollectorまたはExtensionへ引き渡す必要があるため、Fusionやin-place最適化が制約される。`NoCollect`でも観測hookがある限り境界は維持する。

## 3. Dead path elimination

```text
Logical Graph:
 source -> base -> beam
              -> covariance
              -> diagnostic

compile([beam, covariance])

Plan:
 source -> base -> beam
              -> covariance
```

`diagnostic`固有経路は含めない。

## 4. Fusion

直列Node列を一つのFused Nodeへまとめる。

```text
A -> B -> C
```

を

```text
Fused[A, B, C]
```

へ変換する。

目的:

- Python呼び出し回数削減
- 中間buffer削減
- cache locality改善
- C++/Cython backendでのループ統合

Fusion条件:

- 中間Portが観測されない
- 分岐がない
- Extension観測がない
- 時間意味論が互換
- backendが対応
- state境界を壊さない
- frame/rate等のbuffering境界ではない

初期版では安全側に倒し、条件が明白な`map -> map`だけを対象とする。

## 5. Batch化

同一入力を持ち、同一Kernel種別で、parameterだけ異なる並列Nodeをまとめる。

```text
FFT
 ├ BF(el=0)
 ├ BF(el=1)
 └ BF(el=2)
```

を

```text
BatchedBF(el=[0,1,2])
```

へ変換できる。

初期版では候補検出とPlan exportだけ実装し、実変換は後回しでもよい。

## 6. Parallel Stage

依存関係のないNodeを同一Stageへ配置する。

```text
Stage 0: Source
Stage 1: Preprocess
Stage 2: Beam, Covariance, Diagnostic
```

PythonExecutorでは順次実行し、将来のNativeExecutorでは依存関係と決定性契約を満たすStageだけを並列実行可能とする。BackendはStageの並列実行を管理しない。

## 7. Buffer planning

compile時に以下を解析する。

- 各Portの必要滞留数とfan-out consumer cursor
- frame履歴長
- latest保持数
- Source modeとbackpressure可否
- Operation一回あたりのEmission `max_items`
- 出力保持量
- state保持量
- backend workspace
- in-place可否

`PORT_SHARED`はproducer burstを下限に、exact merge分岐が共有する祖先へだけFRAME等の構造需要を逆伝播してPort別capacityを求める。Compilerはcapacity値だけでなく、producer burst、merge Node、分岐需要をPortablePlanIRの`capacity_reasons`へ残す。Python runtimeはこのdescriptor値からPortBufferを生成し、独自に別の上限を再計算しない。

fan-outでは一つの読み取り専用`PortBuffer`を共有し、consumerごとのcursorが通過した時点でitemを解放する。dtype、layout、device変換が必要なEdgeだけ`EdgeAdapterBuffer`を持つ。

静的に決められない場合は、明示`max_items`またはbackpressureを持つbounded dynamic bufferを使う。通常の計算Portは暗黙dropを禁止し、overflow policyを`FAIL`とする。制御不能なリアルタイム入力だけは専用ingress bufferで明示的dropを許可する。完全な分類とdescriptorは[11_Buffer_Scheduler_PortablePlanIR設計.md](11_Buffer_Scheduler_PortablePlanIR設計.md)に定める。

## 8. Scheduler

Schedulerは内部APIとする。

初期版は単一スレッド・決定的Schedulerを採用する。

Python Schedulerは実行可能な参照実装とする。意味論の正本はPlanと本書の
契約である。性能向けには、同じPlanを実行するNativeExecutorを追加し、RATE、
FRAME、buffer、ready判定をnative Stage内で連続実行する。Flow API、Backend、Executorの
責務分離と段階的な実装方針は[10_Native_Executor設計方針.md](10_Native_Executor設計方針.md)に定める。

```text
観測終端から必要intervalを逆伝播
   ↓
ready Node判定
   ↓
不足経路だけを進める
   ↓
Node実行、Emission確定
   ↓
Extensionへ通知
   ↓
観測終端ならcollectorへ配送
   ↓
共有PortBufferへpublish、consumer ready状態更新
```

同一Emissionの配送順は、Extension、観測終端collector、下流consumerの順とする。collector overflowが発生しても、Extensionが安全な劣化結果を先に保存できるようにする。Extension失敗時はcollectorと下流へ配送せず、runを停止する。

SchedulerはSourceを無条件にround-robinで進めない。各Nodeの必要interval、入力cursor、producer frontierを追跡し、未充足需要を生成できる経路だけを進める。exact mergeで必要intervalが生成不能と確定した場合は`STALLED_EXACT_MERGE`を記録し、そのNodeへの需要を停止する。

### 8.1 cooperative Stage実行状態

CppExecutorはImplementation言語にかかわらずPlanを運用できる。現行の
`GraphRuntimeSession.run()`はDAG全体とPort batchをstack-localに保持するため、Python Stageで
中断できない。これをrun-localな継続可能状態へ分離し、実行APIを次に固定する。

```text
advance() -> Completed
          | NeedsPython(stage_id, input_batches)

resume(stage_id, output_batches) -> void
```

`advance()`はreadyなnative Nodeを進め、Python islandの入力が揃った時点でそのみyieldする。
adapterはGIL下でrun-local PythonStageSessionを一回batch実行し、検証済み出力をresumeする。
runtimeはwaiting stage IDとresume stage IDの一致、一回だけのresume、例外後のsession廃棄を保証する。
Stage入出力のinterval、sequence、status、Diagnostic、gap segmentはbatch境界で失わない。

現在のmixed実装は、Python islandの前後にnative区間を置ける。Sourceを含むPython prefixまたは
native prefixのどちらから始まる線形Planでも、複数Python islandをStage順にyield/resumeできる。
`StageDescriptor.input_port_ids/output_port_ids`から境界を特定し、native所有結果をadapterへ、
Python出力を合成native ingressへ、それぞれ一回copyする。Python prefix内のRATE/FRAMEも
Stage単位で処理し、`resume(output_batches)`後はC++ runtimeだけでsuffixを実行する。
native区間からPython islandへ入る複数Portは、`synchronous`の完全interval一致と`latest`の
primary開始時刻以前の最新値をadapterが照合する。Pythonからnativeへの複数ingress、
contains/overlaps/tolerance境界codec、zero-copyは次の継続状態実装で扱う。

Compilerは連続するPython Operationを、観測境界、native依存境界、fan-out ownership境界の
範囲内で最大Python islandへまとめる。全Python Planは可能な限り単一Stageとし、
Python関数ごとやEmissionごとにC++/Python境界を往復しない。

## 9. Push/Pullモデル

Sourceを制御可能性で分類する。

- compile時: 終端から必要経路とbuffer requirementを逆解析
- `PULL_CONTROLLED`: Schedulerがchunk単位でpull requestし、需要またはbuffer余裕がなければ生成を停止
- `REALTIME_PUSH`: device等からの入力をbounded ingress bufferで受け、停止不能な場合だけoverflow policyで棄却

SceneRenderer等は`start_time`と`duration`の要求に対して信号を生成する。

`REALTIME_PUSH`はSource作成時またはbinding時に`max_items`を必須とし、v0.1の既定overflow policyを`DROP_OLDEST`とする。dropは`INPUT_OVERRUN` Diagnosticと欠落intervalを伴う`GapMarker`としてruntimeへ伝える。`BLOCK`はリアルタイムcallbackでは禁止する。

Python Executorは`RealtimeSource.start()`へrun-local `RealtimeIngressBuffer`をreceiverとして渡す。ingressはthread-safeかつnon-blockingで、通常Emissionだけを`max_items`へ数える。dropしたintervalはbuffer容量外のcontrol recordとして連続区間をcoalesceし、次の保持Emissionより前にSchedulerへ渡す。run終了時はSource sessionの`stop()`を必ず呼び、Planへcallback状態を保持しない。

## 10. Runtime Buffer

runtime buffer分類:

- PortBuffer
- EdgeAdapterBuffer
- FrameHistoryBuffer
- LatestStateBuffer
- RealtimeIngressBuffer
- OutputCollector

`FrameHistoryBuffer`と`LatestStateBuffer`は、通常Portのfan-out寿命を持つ`PortBuffer`とは別のrun-local実体とする。Python ExecutorはPortablePlanIRの`owner_node_id`、`owner_input_index`、capacity、reclaim policyから生成し、FRAME履歴やlatest確定値を汎用queueへ暗黙に混在させない。

OutputCollectorは`NoCollect`、`Latest`、`Bounded`、`Sink`のいずれかとし、無制限保持を標準機能にしない。

基本interface:

```python
class PortBuffer(Generic[T]):
    def publish(self, item: StreamItem[T]) -> None: ...
    def peek(self, cursor: ConsumerCursor) -> StreamItem[T] | None: ...
    def advance(self, cursor: ConsumerCursor) -> None: ...
```

## 11. fan-outでの値共有

同一Portから複数Edgeへ出力する場合、NumPy配列等を不要にcopyしない。

```text
immutable StreamItem reference
    ├ consumer A
    ├ consumer B
    └ consumer C
```

consumer cursorで意味論上の寿命を管理する。Executorは同じ解放条件を保つ限り、内部実装に参照カウントを使用してよい。

Kernelは入力を原則として破壊しない。in-place処理はcompileが安全性を証明できる場合だけ許可する。

## 12. ready条件

MAP:
- 必須入力itemが揃う

MERGE:
- 必要logical intervalが全入力で揃う

FRAME:
- 履歴がsize分揃う

RATE:
- 次出力時刻を生成可能

SOURCE:
- Schedulerからrequestがある

REALTIME SOURCE:
- ingress bufferにitemまたはGapMarkerがある

## 13. 停止条件

有限Source:

```python
plan.run()
```

全Source EOFかつ全buffer drain完了で終了する。ただしstalled Nodeしか残らず、観測終端から到達する実行可能な需要がない場合は、制御可能なSourceをEOFまで進めず終了する。この場合はDiagnosticを残し、`RunResult.completed=False`とする。

生成Source:

```python
plan.run(duration=60.0)
```

指定論理時間までSourceを要求し、下流処理をdrainして終了。

`REALTIME_PUSH` Sourceは利用者の停止要求、指定duration、Source sessionのcloseまたはfailで受付を終了し、ingressと下流をpolicyに従ってdrainして終了する。

終了前にcollectorをcloseし、Extensionのbounded queueをpolicyに従ってflushしてから`finalize`を呼ぶ。

## 14. Planの再利用

PlanとKernelはSession間で共有可能な不変情報を保持する。Extension観測も`ObservationSpec`として
Planへ固定するが、handler実体と可変状態は保持しない。`Plan.create_session()`がprocess-local bindingを
検証し、Kernelの`create_state()`とExtensionの`create_session()`からrun-local状態を生成する。
v0.4の`CompiledKernel`はKernelへのdeprecated aliasである。

観測契約がないPlanでは`plan.run()`を`plan.create_session().run()`の簡略形として使用できる。必須Extension bindingがあるPlanで`plan.run()`を呼んだ場合は、binding不足を暗黙に無視せず明示例外にする。

初期版推奨:

- 各`run()`開始時に新しいsessionを生成
- 同じSessionを再実行してもExtension triggerとhandler状態をreset
- v0.2の継続実行は`Plan.create_continuous_session()`で明示的に生成
- `ContinuousSession.start()`から`close()`または`cancel()`までKernelState、FRAME、RATE、buffer、collector、Extension trigger状態を保持
- `run_until(logical_end)`は終端が指定境界以下のEmissionまで進め、境界外の既取得Emissionをsession内に保留
- `run_until()`が返す`RunResult`はsession開始時からの累積snapshotとし、別sessionへ可変状態を持ち越さない

v0.2ではfinite Sourceに加え、RealtimeSourceの停止、fail、ingress drainを実装する。`close()`はSource sessionへstopを一度だけ要求してingressをcloseし、残るGapMarkerとEmissionを下流までdrainする。`cancel()`はingressをdiscardし、Node別破棄件数をDiagnosticへ残す。非finite pull SourceはEOFを推測せず、`close()`では明示例外として`cancel()`を要求する。

`RuntimeOptions`はSource request chunk、PORT_SHARED high/low watermark、呼出し単位のScheduler budget、profiler有効化をConfigと分離して持つ。budget到達は`EXECUTION_BUDGET_EXHAUSTED`として結果へ残し、同じ論理境界から継続できる。

Profilerは無効時に時計を読まず、`RunResult.profile=None`とする。有効時はScheduler step、Kernel call/時間、buffer current/high-watermark、Source emission/pending/drop/logical endをsession-local snapshotとして返す。値、interval、sequence、status、Diagnosticの配送順は変えない。

PortablePlanIR schema 0.2はNode parameter、同期tolerance/missing policy、adapter buffer、callable gap policyを追加する。readerは0.1のoptional field欠落を受理し、FRAME/RATE parameterをtime descriptorから復元する。`bind_plan()`はSource、Operation implementation、collector、Extension slotとConfig scopeの完全一致を検証してからGraphを再構築する。legacy schemaのKernel bindingはOperation adapterへ正規化する。

## 15. 決定性

同じGraph、入力、Config、backend、optimizationなら可能な限り同じ実行順と結果を保証する。

並列backendによる浮動小数演算順の差は許容範囲を文書化する。

## 16. 劣化結果の配送

SchedulerはEmission statusを保持したままbuffer、後段Node、collector、Extensionへ配送する。`DEGRADED`または`INVALID`はready判定上も一つのEmissionであり、暗黙にdropしない。

後段Kernelが受理しないstatusを受け取った場合は、Kernel宣言に従って以下のいずれかを行う。

- statusを伝播する
- 安全な別fallbackを生成する
- `INVALID`を生成して処理を継続する
- 安全に継続できない場合だけ例外にする
