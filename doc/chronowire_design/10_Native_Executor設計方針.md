# Native Executor設計方針

本書のv0.4 `Kernel`/`KernelState`表記は現実装を説明する。今後の公開Operation、native
module manifest、ImplementationBindingとの対応は[14_Operation設計.md](14_Operation設計.md)を正本とする。

## 1. 目的

Chronowireは、利用者がPythonのFlow APIで処理の流れだけを記述し、そのLogical Graphをcompileして実行する。高速化のために公開Flow APIをCythonまたはC++へ置き換えるのではなく、反復実行されるPlanのExecutorをnative化する。

この分離により、次を同時に満たす。

- Pythonでは読みやすいFlow APIを維持する
- Python、Cython、C++ Kernelで同じLogical Graphと論理時間契約を共有する
- Backend変更時もNode依存関係、rate、frame、collectorの意味を変えない
- Python callbackを含まないStageでは、EmissionごとのPython dispatchを避ける
- 最適化前後で値、interval、sequence、status、Diagnosticを保存する

## 2. 責務分離

```text
Python Flow API
    ↓ Logical Graph構築
Compiler
    ↓ PortablePlanIR
Binding
    ↓ BoundPlan
Executor
    ├ PythonExecutor
    └ NativeExecutor
          ↓
       KernelState
```

### 2.1 Flow API

Flow APIはPythonに残す。Flow構築は実行前に一度だけ行われるため、主要な性能対象ではない。

責務:

- Node、Port、Edgeの宣言
- Config scopeの指定
- 同期Flow、StateFlow、rate、frameの記述
- 観測対象Flowの指定

collector、Extension、Backend、Executorの選択はFlow chainではなく、`compile()`またはPlan bindingの責務とする。

### 2.2 Backend

Backendはoperation IDに対応する`ImplementationSpec`を明示選択し、内部`Kernel`へ変換する。アルゴリズムの実装言語、SIMD方式、FFT plan、係数、workspace要件を決定する。v0.4ではこのfactoryを`Kernel`と呼ぶ。

BackendはNodeの起動順、rate cursor、Edge buffer、fan-out寿命を管理しない。

### 2.3 Executor

ExecutorはPlanを実行する。

責務:

- Source request
- ready Node判定
- rate cursor
- frame history
- FIFO、ring、latest buffer
- Node起動
- fan-out参照寿命
- collectorとExtension境界への配送
- run-local `KernelState`の生成と所有

BackendとExecutorを分離し、`CppBackend + PythonExecutor`と`CppBackend + NativeExecutor`の両方を可能にする。前者は機能互換性の確認、後者はSchedulerを含む高速化に使う。

### 2.4 Python control planeとC++ data plane

ChronowireはPythonを実行系から排除しない。Pythonは利用者が処理を構築、検証、起動、観測する
control planeとして残し、繰り返し実行だけをC++ data planeへ移す。

| plane | 所有する責務 |
|---|---|
| Python control plane | Flow構築、OperationSpec/Config宣言と検証、BackendとImplementation選択、compile、PortablePlanIR生成、module load/bind、session生成、run/flush/close/cancel起動、collector/Extension設定、Diagnosticと計測結果の公開 |
| C++ data plane | run開始後のSource ingress、論理時間Scheduler、RATE/FRAME/SAMPLE、Edge buffer/cursor/fan-out寿命、KernelState、native Operation呼出し、status/Diagnostic伝播、collector保持選択、実行計測 |

control planeからdata planeへの呼出しはcompile、session生成、run等のライフサイクル境界に
限定する。完全native PlanではEmissionごとにPythonがNode起動、ABI dispatch、buffer変換を行わない。
Pythonを通るcollector、Extension、plain callableを含む場合は、compile時に明示されたPython Stage
境界として分割する。「Python-free hot path」は「Python-free application」を意味しない。

## 3. 性能上の基本方針

### 3.1 blockを実行単位にする

1 sampleを1 EmissionとしてPython Executorへ通す構成を性能目標にしない。信号処理Kernelは原則として、数百から数千sampleのblockを一つのnative bufferとして受け取る。

Python Executorを使用する場合も、block単位にすることでPython dispatchを償却する。

ここでいうblockは物理的な実行batchであり、論理的なEmission単位とは区別する。v0.1の`frame(size=N)`はN個の論理itemを集める契約を維持する。NativeExecutorは複数の論理itemを`ItemBatch`として一括保持・処理し、RATEとFRAMEをvectorizeしてよいが、interval、sequence、status、Diagnosticの意味を変えてはならない。

### 3.2 native StageではPythonへ戻らない

NativeExecutorが連続実行するStageでは、次を禁止する。

- EmissionごとのPython callback
- Pythonのdataclass、tuple、dictによる内部queue
- sampleごとのGIL取得
- native bufferからPython objectへの不要な変換

Extension、Python collector、Python callableなど、利用者が明示した観測・実行境界でのみPythonへ戻る。

### 3.3 NumPyと同じ呼出し粒度を目指す

NumPyと同様に、Pythonは処理を一度起動し、反復loopはnative領域で完了させる。Flow APIをnative化すること自体ではなく、一回のPython呼出しで十分な処理量をnative実行することを重視する。

### 3.4 完全native hot pathの成立条件

完全native Stageでは、Emission数に比例するPython callback、Python object生成、tuple/dict変換、
属性参照、reference count更新、GIL取得を許可しない。Pythonのloaderがcontrol planeで
module tableとfunction pointerを解決することは許可するが、itemごとに`ctypes`等から呼び出さず、
C++ runtimeが解決済みfunction pointerを直接呼ぶ。

Pythonの公開`Emission`はRunResult、Python collector、Extension、Python Stageの境界でのみ復元する。
`NoCollect`またはnative Sinkでは復元しない。`Latest`/`Bounded`ではC++側で保持対象を選択し、
Python object生成量を保持件数でboundする。

成立はコードの見た目ではなく計測で判定する。完全native Planの受入条件は次とする。

- native Stage内のPython Stage dispatch数が0
- Python/nativeライフサイクル遷移数がEmission数に依存せず、runあたり定数回
- `run()`中のC++ Scheduler/Operation loopがGILを保持しない
- `NoCollect`とnative Sinkのoutput boundary copy byteが0
- Python object生成数が入力Emission数ではなく公開保持件数でboundされる
- Python Stageを含む場合、その境界数、batch変換量、GIL取得回数がPlanとprofilerで可視化される

## 4. portable Plan

Plan表現を次の三層へ分ける。

```text
Logical Graph
    ↓ compile
PortablePlanIR
    ↓ bind
BoundPlan
```

- `PortablePlanIR`: Python objectを含まないserialization可能な意味論上の計画
- `ExecutionBindings`: Source、ImplementationBinding、plain Python callback、collector、Extensionのprocess-local binding
- `BoundPlan`: PortablePlanIRとExecutionBindingsをExecutorへ渡す組

既存の公開`Plan`は両者を包む。Source、Operation implementation、collectorのprocess-local bindingはPlan生成時に確定できるが、Extension handlerは`Plan.create_session(extension_bindings=...)`で明示注入する。`extension_id`は観測契約の安定ID、binding slotはprocess-local注入機構であり同一fieldにしない。Native pointer、FFT plan handle、Python callable、shape resolver、module path、Extension path/objectそのものはPortablePlanIRへ保存しない。

NativeExecutorへ渡す`PortablePlanIR`は、Python object graphを参照せず、次の固定情報を持つ。

- schema version
- Node、Port、Edge ID
- topological orderまたはStage
- Node opcode
- input semantics
- integer ticksとrational timebase
- rate period
- frame size、hop、EOF policy
- buffer種別、capacity、ownership
- operation ID、resolved schema、選択implementation/ABI IDとworkspace descriptor
- collector、Extension、Python callback境界
- ExtensionDescriptorのID、観測Port、論理時間trigger、priority、policy、binding slot、ABI
- Backend fallback policy

RATE、FRAME、buffer管理はExecutor opcodeとして表し、DSP Kernelへ混ぜない。

Node、Port、Edge、Buffer、Time、Source、Binding descriptorの必須field、共有`PortBuffer`のconsumer cursor、Source mode、frontier、realtime gapの意味論は[11_Buffer_Scheduler_PortablePlanIR設計.md](11_Buffer_Scheduler_PortablePlanIR設計.md)を正本とする。

## 5. native StreamItemとbuffer

Native Stage内では、Pythonの`Emission`に相当する情報を固定構造で保持する。

```cpp
struct NativeStreamItem {
    BufferView value;
    LogicalTime start;
    LogicalTime end;
    uint64_t sequence;
    EmissionStatus status;
    DiagnosticHandle diagnostics;
    MetadataHandle metadata;
};
```

`BufferView`は少なくともdtype、shape、stride、device、read-only、ownershipを明示する。fan-outでは一つの`PortBuffer`にimmutable参照を保持し、consumer cursorで意味論上の寿命を管理する。native実装は同じ解放条件を保つ限り参照カウントを併用できる。

Python境界では`NativeStreamItem`から公開`Emission`へ変換する。native Stage内部では変換しない。

### 5.1 Kernel ABIの最小契約

以下はv0.4 Kernel ABIの成立条件である。Operation移行では、C++/Cython sourceへdecoratorを書かず、
version付きC ABI module tableがoperation ID、implementation ID、create/process/flush/destroyを提供する。
symbolic shapeはOperationSpecでcompile時に解決し、native moduleはresolved schemaとBufferViewだけを
再検証する。詳細は[14_Operation設計.md](14_Operation設計.md#8-native-moduleとc-abi)を参照する。

v0.4では`NativeOperationModule`が明示された共有libraryをloadし、module ABI、entry size、重複ID、
function table、flush flag、alignmentを検証する。`NativeModuleBackend`はConfigをimmutable parameterへ
固定し、CppExecutorはfunction addressからrun-local sessionをcreate/process/destroyする。moduleが返した
statusとDiagnosticはC++内でcopyし、collector境界でNode、Port、interval付き公開Diagnosticへ戻す。
共有libraryのpathとhandleはPortablePlanIRへ含めず、別processではImplementationBindingを再注入する。

Kernel ABIは少なくともversion、session生成、process、flush、session破棄、workspace requirement、error codeを持つ。C++例外をABI境界外へ出さず、失敗はerror codeとDiagnosticへ変換する。

compile時にはworkspaceのsize、alignment、device要件だけを決定し、実bufferはrun-local sessionが確保・解放する。PortablePlanIRにはABI IDとdescriptorを置き、pointerやallocator instanceはExecutionBindingsへ置く。

## 6. Stage分割

compile時にNodeを次の境界でStageへ分割する。

- Python callable
- Python Source、collector、Extension
- Backendまたはdevice変換
- dtype、layout変換
- 観測境界
- native化されていないopcode

例:

```text
Native Stage: Source → RATE → FRAME → FFT → CBF
Python Stage: user_callback
Native Stage: FIR → native Sink
```

Python callableを含むPlanも実行可能とするが、Python境界数と変換量をPlan exportおよびDiagnosticへ記録する。

Python callableはcompile時に`PYTHON_CALLBACK`または`PythonCallableKernel`へ正規化し、PortablePlanIRにはcallback binding IDだけを記録する。Native Stageとの境界ではbatch変換、GIL取得、0/1/複数Emissionの正規化、例外位置を明示する。

Operation設計では選択BackendにImplementationがなければ既定でcompile errorとし、Python実装へ暗黙に
fallbackしない。将来`PreferNative`等の明示policyを追加した場合だけ、fallback理由とPython Stage境界を
DiagnosticおよびPlanへ記録する。v0.4のSTRICT/DEFAULT表記を新Operationの既定へ引き継がない。

## 7. CythonとC++の位置づけ

Python、Cython、C++は全Operationが必ず通過する直線的な移植手順ではない。OperationSpecと
Plan契約から意味論とgoldenを固定し、profileで支配的と確認した境界だけを
CythonまたはC++へ移す。PythonExecutorは参照実装だが、意味論の正本ではない。
CythonはPython runtimeを保ったままtyped buffer、`nogil`、C ABIを実証する中間手段であり、
C++ first OperationはCython実装を経由しなくてよい。逆にCythonで目標性能と配布性を満たす
OperationをC++へ機械的に書き換えない。

### 7.1 Cython Executor

最初の設計実証に使用する。

- Python実装との比較が容易
- typed memoryviewでnative bufferを試せる
- `nogil`でrate、frame、ready判定loopを検証できる
- 現行Planから必要なIR項目を抽出できる

Cython固有表現をportable Planの正本にはしない。

### 7.2 C++ Executor

長期的な標準NativeExecutor候補とする。

- Python以外のbindingを追加できる
- thread pool、SIMD、real-time制約へ発展できる
- Cython KernelとC++ Kernelを共通ABIへ接続できる
- Schedulerとbuffer ownershipをPython runtimeから分離できる

### 7.3 同一Operationの段階的最適化

同一のFlowとOperationSpecに対し、次の実装形態を交換可能にする。

1. OperationSpecと論理時間/status/Diagnostic契約からgolden traceと不変条件を固定し、
   Python Implementation + PythonExecutorを実行可能な参照実装として検査する。
2. 必要な場合だけCython Implementationでhot loopを高速化し、PythonExecutorまたはCythonExecutorで比較する。
3. C ABI wrapperの背後にCythonまたはC++ DSP実装をbindし、C++ Backend + PythonExecutorでOperation単体を照合する。
4. C++ Backend + CppExecutorでcompile後PlanのScheduler、buffer、KernelStateを含むdata plane全体をnative運用する。

段階を変えてもFlow、OperationSpec、Config scope、logical timeは書き換えない。各段階は
契約由来のgolden traceに対する値、interval、sequence、status、Diagnostic、gap/resetの同値性と、
境界ごとの計測をgateとする。

## 8. streaming CBFによる設計実証

最初の比較対象はstreaming CBFとする。

同じFlowとPlan意味論に対して、次を比較する。

1. Python Executor + Python Kernel
2. Python Executor + native CBF Kernel
3. 最小Cython Executor + native CBF Kernel

SourceからExecutorへの物理入力はchannel-first blockとする。block内の各sampleは論理itemとして扱い、Executor内部の`ItemBatch`からrate、frame history、EOF padding、CBF、collectorへ一括配送する。

確認項目:

- 値、interval、sequenceの一致
- `DEGRADED`、`INVALID`、Diagnosticの一致
- fan-out時の二重実行がないこと
- 同じPlanを再実行したときのsession分離
- Python callback回数
- native/Python境界数
- block size別throughputとlatency
- copy回数と最大buffer量

v0.3のBackend交換conformanceとして、固定CBF宣言をPython実装とCython `nogil`実装で
個別にcompileし、さらにPython前処理、Cython CBF、Python後処理を一つのPlanへ配置する。
参照DSPコードは`chronowire_reference` packageへ分離し、Chronowire本体の公開Flow APIへ
CBFを追加しない。これらにCython Executor + Cython CBFを加えた四構成は、同じ値、interval、
sequence、status、Diagnostic traceを要求する。

`0.4.0`では、同じ固定shape PlanをC++ Executorでも実行する五つ目の構成を追加する。Cython
Executor間の相互比較だけでなく、契約から固定したgolden traceと各Executorを独立に比較し、
C++への移行で意味論を変えない。

`f64_vector_source`経路ではSource値を固定channel shapeとして宣言し、Cython Executorが
RATE/FRAMEをnative batch化する。frame batchはread-only contiguous f64 memoryviewとして
`NativeBatchKernelState.run_batch()`へ一回だけ渡し、CBF出力もnative bytes batchで返す。
Python tupleへの復元はcollector境界だけで行う。現実装にはCython Schedulerからbatch Kernelを
起動するStage単位のPython method callとbuffer copyが一回残るため、copy回数とlatencyを測定して
C ABI pointer呼出しへ進むか判断する。

測定方法と2026-07-22時点の結果は
[13_CppExecutor移行測定.md](13_CppExecutor移行測定.md)を正本とする。8192 sample、4 channel、
2 beamでは、Cython RATE/FRAMEとCBF callの合計はend-to-end中央値の1%未満で、Source Emission化と
論理時間/native array packingが約87%から89%を占めた。したがってC++化はCBF loopの書き換えを
先行せず、owned C ABI、prepacked native ingress、観測境界まで遅延するEmission復元を一つの
Executor責務として実装する。

## 9. 実装段階

Native Executorはv0.2の必須範囲には含めない。[12_v0.1_v0.2リリース方針.md](12_v0.1_v0.2リリース方針.md)のv0.1/v0.2 release gateを満たした後、v0.3以降で以下のPhaseを開始する。

### Phase 1: block粒度の基準測定

- 現行Python Executorでblock単位CBFを実行する
- sample単位版とのdispatch回数と処理時間を比較する
- 値と論理時間の基準結果を固定する

v0.3開始時点で`examples/native_executor_baseline.py`にsample粒度と4-sample block粒度の固定CBFを実装した。値、interval、sequence、status、Diagnosticの一致をtestし、Kernel呼出し回数、Scheduler step、Kernel実行時間、buffer high-watermarkを意味論traceと分離して測定する。wall-clock値は環境依存なのでgolden値にはしない。

### Phase 2: conformance traceと最小IR/ABI

- RATE、FRAME、status、Diagnostic、metadata、sequenceの基準traceを固定する
- SOURCE、RATE、FRAME、単一MAP、collectorに限定したPortablePlanIRを定義する
- BufferView、ItemBatch、Kernel ABIの実験版を定義する
- Extension、collector、consumerの配送順を固定する

PortablePlanIR schema 0.3は最小`StageDescriptor`、`ValueSchemaDescriptor`、実験的`KernelAbiDescriptor`を追加する。現Python Portは`python_opaque`、現Kernelは`python-v1`かつnative非互換、workspace未宣言と記録する。未確定情報をNative Executorが推測してはならない。

`Plan.run()`、`create_session()`、`create_continuous_session()`は`Executor`を選択でき、既存runtimeを`PythonExecutor`として使用する。これはExecutor差し替え境界の固定であり、まだNative実行を意味しない。

Operation implementationの選択はcompile時に`backend`既定selectorとoperation IDごとの
`implementations` overrideで確定し、Executorはrun時に独立して選ぶ。PythonExecutorはPython
Implementationだけでなく、選択済みC ABI ImplementationをEmission単位で直接呼べるため、native
Operationのconformanceとデバッグに使う。CppExecutorは全対象Stageがnative対応の場合に同じbindingを
C++ runtimeから直接呼ぶ。Python Implementationを含む場合はPython Stage runnerへ明示的に
yieldする。これはfallbackではなくPortablePlanIRに固定されたStage実行である。

新規DSP Operationや既存ABI上のImplementation追加はDSP package側で完結させ、Executorを
改修しない。Python/Cpp Executorの両方を改修するのは、Scheduler、buffer、時間、
status/Diagnostic、lifecycle、ABI/process modelなどPlanの実行意味論を拡張するときである。
この二重実装は独立Executorの契約一致を保つための必要経費とする。段階導入は認め、
未対応capabilityはNode、Port、binding、違反契約を伴う明示エラーで拒否する。

### Phase 2.5: cooperative Python Stage

C++ runtimeのDAG実行状態をstack-localな`run()`からrun-local継続状態へ分離する。

1. `advance()`が`Completed`または`NeedsPython(stage_id, input_batches)`を返す
2. adapterがGIL取得後に最大Python islandの`PythonStageSession`をbatch実行する
3. schema、0/1/複数Emission、時間、sequence、status、Diagnosticを境界で検証する
4. `resume(stage_id, output_batches)`が出力batchのownershipを確定してSchedulerを再開する

C++ runtime内にPython C API、`PyObject*`、callback function pointerを持ち込まない。固定schemaは
read-only memoryviewでborrowし、適合するbuffer出力はzero-copy、それ以外は境界ごと1回copyとする。
`python_opaque`を扱うRATE/FRAMEは初期実装でPython islandに含めてよい。

初期実装はall-Python one-shot Planに加え、Python islandの前後へnative区間を置くmixed Planを扱う。
Python/nativeどちらのprefixから始まる線形Planでも複数Python islandをStage順に実行する。
C++ GraphRuntimeSessionがnative区間を自立実行し、終端なしcollectorで
Stage入力batchを所有結果としてadapterへ渡す。adapterはstage IDの`advance()` / `resume()` /
`abort()`状態に従い、run-local Python session群をStageにつき一回dispatchする。Python出力は
shape/time/statusを検証して合成SOURCE ingressへ一回copyし、C++ suffixを再開する。fan-out、
0/複数Emission、RATE/FRAME、status/Diagnostic、例外後の再実行を照合済みである。
native→Python複数入力は`synchronous`完全interval一致と`latest`選択を実装済みである。
Python→native複数ingress、contains/overlaps/tolerance境界codec、zero-copy、mixed ContinuousSessionは
未対応capabilityとしてStage/Node/Port/binding付きで拒否する。all-native hot pathは変更しない。

### Phase 3: 最小Cython Executor

- SOURCE、RATE、FRAME、単一native MAP、collector終端に限定する
- 同じPortablePlanIR意味論から実行する
- Stage内ではPython objectを生成しない
- Python Executorとの意味論一致を試験する

`0.3.0.dev0`の最初の実証では、有限`f64_source`、`RATE(HOLD)`、
`FRAME(pad_end=False)`、`identity_f64` MAP、一つのcollector終端に対象を限定した。
RATEとFRAMEは有理数periodを共通分母の整数tickへ変換し、native配列上の`nogil` loopで
実行する。Pythonの`Emission`はcollector境界でだけ構築する。

この限定経路では値、interval、sequence、status集計、空入力、重複frameを
Python Executorと比較する。native itemはstructure-of-arraysとし、論理時間は共有有理
timebase上のsigned i64 tick、statusはu8、DiagnosticはSource provenance indexとして運ぶ。
collector境界でだけ公開`Emission`とDiagnostic列を復元するため、同じSource Diagnosticが
RATEで複製された場合の重複と順序も保存できる。DEGRADEDは値を保持し、INVALIDを受理しない
MAPではPython Executorと同じ`INVALID_INPUT_PROPAGATED`を生成する。

Python値、任意callback、Extension、EOF padding、`ContinuousSession`は未対応である。
`INPUT_OVERRUN`はRATE cursorだけでなくFRAME履歴もresetする必要があるため、この段階では
`contract=gap_reset`として拒否する。未対応契約はPythonへ暗黙fallbackせず、契約名、Node、
Portを含むエラーにする。汎用metadata table、gap-aware frame state、native CBFはPhase 4で接続する。

### Phase 4: portable IRとnative buffer契約の固定

- Plan schemaを固定する
- Kernel ABI、BufferView、ownershipを定義する
- Python境界とfallbackをPlanへ記録する

### Phase 5: C++ Executor

- Cython実証で確定した状態機械をC++へ移す
- owned C ABIでScheduler/Kernel境界copyとPython dispatchを除去する
- prepacked integer-tick ingressとnative Sink/NoCollectを性能経路にする
- parallel Stage、thread pool、native Sinkは測定後に追加する

`0.4.0`では、PortablePlanIRの`SOURCE`、`RATE`、`FRAME`、`MAP` descriptorとprocess-local
Source/Kernel bindingからrun-local C++ DAG sessionを生成する。sessionはprepacked f64値、signed i64
source tick、status、reset境界、登録済みKernel定数を所有し、複数native MAP、fan-out、複数output、
NoCollect/Latest/Boundedの保持選択をGILなしで実行する。NoCollectでは出力値をPython境界へcopyせず、
Bounded/Latestでは保持対象だけを公開Emissionへ復元する。

Kernel dispatchはversion付きABI resolver tableでidentity f64、固定CBF、および実数MVDR参照ABIを選択する。
この登録済みtableはOperation固有知識をCppExecutorに持つ過渡的実装である。新規Operationは
既に実装した外部C ABI module tableの汎用dynamic bindingを使い、旧tableは段階的に置換する。
Operation追加ごとにCppExecutorのresolverを増やさない。fan-outでは親batchを
read-only共有し、topological run内で各Nodeを一度だけ評価する。`INPUT_OVERRUN`のsegment境界でRATE
cursorとFRAME履歴をresetし、INVALID/DEGRADED、metadata、Diagnostic provenanceを出力まで保つ。
有限`ContinuousSession`は排他的論理時間ごとの決定的snapshot、flush、close、cancelを提供する。one-shot
Extensionは観測PortをC++ collectorとは別に全件取得し、C++ run終了後のPython Stage境界でtriggerと
priorityを適用する。

周期MVDR受入経路では、`FRAME`から累積共分散を生成し、`SAMPLE` Nodeで安定した整数論理時間境界だけを
重み更新へ渡し、`LATEST`入力として各frameへ適用する。C++ DAG runtimeは複数入力、latest保持、run-local
共分散状態を所有する。初期積分不足は例外にせず`DEGRADED`とDiagnostic provenanceをbeamまで伝播する。
PythonExecutorとCppExecutorで同じPlanを実行し、値、interval、sequence、status、Diagnostic、およびrun間
resetを比較する。この実数参照ABIはPlan運用のconformance用であり、Chronowire本体へ本番用MVDR DSPを
内包するものではない。

単一有限f64 vector Source、`FRAME(pad_end=False)`、登録済みABI、既定RuntimeOptionsが
v0.4のportable範囲である。realtime push、複数Source/merge、manifest由来の汎用mutable native workspace、incremental
cursor、継続Extension sessionはv0.5へ送り、v0.4では暗黙fallbackせず明示エラーにする。意味論と
再測定結果は[13_CppExecutor移行測定.md](13_CppExecutor移行測定.md)を正本とする。

## 10. 完了条件

Native Executor設計は、次を満たした時点で成立とする。

- 同じFlow記述からPythonExecutorとNativeExecutorを選択できる
- Scheduler操作を公開Flow APIへ露出しない
- RATE、FRAME、Kernelの意味がExecutor間で一致する
- native Stage内にEmissionごとのPython dispatchがない
- Python境界とfallbackがcompile時に判別できる
- degraded/invalid結果とDiagnosticが高速化で失われない
- streaming CBFで値の一致と性能差を再現可能に報告できる

Native Executor実装へ進む前に、Python ExecutorでRATE、FRAME、配送順、status、Diagnostic、session分離のconformance testが成功していることを必須gateとする。

## 11. 非目標

- Flow API全体のCython/C++化
- Chronowire本体へのFFT、CBF、MVDR実装の内包
- 最初から全Nodeをnative化すること
- benchmarkなしのparallel化またはin-place最適化
- Python callbackを暗黙にnative化すること
