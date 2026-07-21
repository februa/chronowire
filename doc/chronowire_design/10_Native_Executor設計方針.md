# Native Executor設計方針

## 1. 目的

Chronowireは、利用者がPythonのFlow APIで処理の流れだけを記述し、そのLogical Graphをcompileして実行する。高速化のために公開Flow APIをCythonまたはC++へ置き換えるのではなく、反復実行されるExecutionPlanのExecutorをnative化する。

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
    ↓ BoundExecutionPlan
Executor
    ├ PythonExecutor
    └ NativeExecutor
          ↓
       CompiledKernelSession
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

BackendはKernelを`CompiledKernel`へ変換する。アルゴリズムの実装言語、SIMD方式、FFT plan、係数、workspace要件を決定する。

BackendはNodeの起動順、rate cursor、Edge buffer、fan-out寿命を管理しない。

### 2.3 Executor

ExecutorはExecutionPlanを実行する。

責務:

- Source request
- ready Node判定
- rate cursor
- frame history
- FIFO、ring、latest buffer
- Node起動
- fan-out参照寿命
- collectorとExtension境界への配送
- run-local `CompiledKernelSession`の生成と所有

BackendとExecutorを分離し、`CppBackend + PythonExecutor`と`CppBackend + NativeExecutor`の両方を可能にする。前者は機能互換性の確認、後者はSchedulerを含む高速化に使う。

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

## 4. portable ExecutionPlan

Plan表現を次の三層へ分ける。

```text
Logical Graph
    ↓ compile
PortablePlanIR
    ↓ bind
BoundExecutionPlan
```

- `PortablePlanIR`: Python objectを含まないserialization可能な意味論上の計画
- `ExecutionBindings`: Source、CompiledKernel、Python callback、collector、Extensionのprocess-local binding
- `BoundExecutionPlan`: PortablePlanIRとExecutionBindingsをExecutorへ渡す組

既存の公開`ExecutionPlan`は両者を包む。Source、Kernel、collectorの既存Python bindingはPlan生成時に確定できるが、Extension handlerは`ExecutionPlan.create_session(extension_bindings=...)`で明示注入する。`extension_id`は観測契約の安定ID、binding slotはprocess-local注入機構であり同一fieldにしない。Native pointer、FFT plan handle、Python callable、Extension path/objectそのものはPortablePlanIRへ保存しない。

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
- Kernel ABI IDとworkspace descriptor
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

STRICT modeではnative化できないNodeをcompile errorとする。DEFAULT modeではPythonExecutorへのfallbackを許可し、warningを残す。

## 7. CythonとC++の位置づけ

### 7.1 Cython Executor

最初の設計実証に使用する。

- Python実装との比較が容易
- typed memoryviewでnative bufferを試せる
- `nogil`でrate、frame、ready判定loopを検証できる
- 現行ExecutionPlanから必要なIR項目を抽出できる

Cython固有表現をportable ExecutionPlanの正本にはしない。

### 7.2 C++ Executor

長期的な標準NativeExecutor候補とする。

- Python以外のbindingを追加できる
- thread pool、SIMD、real-time制約へ発展できる
- Cython KernelとC++ Kernelを共通ABIへ接続できる
- Schedulerとbuffer ownershipをPython runtimeから分離できる

## 8. streaming CBFによる設計実証

最初の比較対象はstreaming CBFとする。

同じFlowとExecutionPlan意味論に対して、次を比較する。

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

`ExecutionPlan.run()`、`create_session()`、`create_plan_session()`は`Executor`を選択でき、既存runtimeを`PythonExecutor`として使用する。これはExecutor差し替え境界の固定であり、まだNative実行を意味しない。

### Phase 3: 最小Cython Executor

- SOURCE、RATE、FRAME、単一native MAP、collector終端に限定する
- 同じPortablePlanIR意味論から実行する
- Stage内ではPython objectを生成しない
- Python Executorとの意味論一致を試験する

`0.3.0.dev0`の最初の実証では、有限`f64_source`、`RATE(HOLD)`、
`FRAME(pad_end=False)`、`identity_f64` MAP、一つのcollector終端に対象を限定した。
RATEとFRAMEは有理数periodを共通分母の整数tickへ変換し、native配列上の`nogil` loopで
実行する。Pythonの`Emission`はcollector境界でだけ構築する。

この限定経路では値、interval、sequence、OK status集計、空入力、重複frameを
Python Executorと比較する。Python値、任意callback、Extension、EOF padding、
`PlanSession`、DEGRADED/INVALIDとDiagnosticを伴う入力は未対応であり、Pythonへ
暗黙fallbackせず契約名、Node、Portを含むエラーで拒否する。status/Diagnosticを含む
native item ABIとnative CBFはPhase 4の契約固定後に接続する。

### Phase 4: portable IRとnative buffer契約の固定

- ExecutionPlan schemaを固定する
- Kernel ABI、BufferView、ownershipを定義する
- Python境界とfallbackをPlanへ記録する

### Phase 5: C++ Executor

- Cython実証で確定した状態機械をC++へ移す
- parallel Stage、thread pool、native Sinkは測定後に追加する

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
