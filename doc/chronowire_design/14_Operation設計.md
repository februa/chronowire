# Operation設計

## 1. 位置付け

本書は、Flow境界で実行する利用者処理の公開契約を`Operation`として定める正本である。
公開語彙は`Flow → compile() → Plan → create_session() → Session`、処理部品は
`Operation → Kernel → KernelState`とする。v0.4のcompile前`Kernel` protocolは内部
`KernelProvider`へ移し、`NativeValueSchemaProvider`のshape責務はOperationSpecへ集約する。

Operation導入後もChronowireの中心は次のままである。

- 利用者はPythonでFlowとして処理の流れを記述する
- Flow構築時には処理を実行しない
- compileがLogical Graphを検証してPortablePlanIRを生成する
- BackendはOperationの実装を選択・compileする
- Executorはcompile後のScheduler、Edge buffer、collector、run-local sessionを運用する
- FFT、FIR、CBF、共分散、resampling等のアルゴリズムは外部DSP packageが提供する

Operationは物理bufferを公開APIへ露出する仕組みではない。`func(inputs, config)`の`inputs`は、
主入力、同期入力、latest StateFlow入力を名前で参照する不変な論理入力集合である。連結領域、batch、
stride、SIMD幅等は選択実装とExecutorの責務であり、Flow利用者は意識しない。

## 2. 用語と責務

公開名は次の規則へ統一する。似た責務を多数の`*Contract` classへ分割しない。

| 接尾辞 | 意味 | 代表例 |
|---|---|---|
| `Spec` | compile前の言語非依存宣言 | `OperationSpec`、`ValueSpec`、`ConfigSpec` |
| `Descriptor` | compile後のportable情報 | `OperationDescriptor` |
| `Binding` | process-localな実体 | `ImplementationBinding` |
| `Session` | Plan全体の実行入口とrun-local状態 | `run`、`start/run_until/close` |
| `KernelState` | Kernelごとのrun-localな能動的実行主体 | `process`、任意の`flush/close` |

中心となる公開宣言は次の五つとする。

- `OperationSpec`: Operation全体の意味論
- `OperationInputSpec`: 名前、primary/synchronous/latest、required、ValueSpec
- `OperationOutputSpec`: 名前、ValueSpec、時間、status、Emission件数
- `ValueSpec`: dtype、item shape、device、representation、read-only
- `ConfigSpec`: scope内の必須leaf、型、任意の制約

compile後は`OperationDescriptor`と、選択された`ImplementationSpec`のportable部分をPlanへ記録する。
実装module、function pointer、Python callableは`ImplementationBinding`へ置く。内部compile済みfactoryは
`Kernel`、実行中の履歴、workspace、native handleは`KernelState`が所有する。

## 3. 公開Flow API

### 3.1 基本形

利用者が処理を接続するAPIは従来どおり`flow.map(func)`とする。

```python
filtered = signal.map(fir)
```

receiverである`signal`は`OperationSpec`が指定するprimary inputへbindされる。追加Flowは入力名で
bindする。

```python
combined = signal.map(
    combine,
    reference=reference,
    calibration=calibration_state,
)
```

この例の論理入力集合は概念的に次となる。

```text
inputs["signal"]       primary synchronous input
inputs["reference"]    named synchronous input
inputs["calibration"]  named latest StateFlow input
```

入力名、種類、必須性は`OperationInputSpec`で宣言する。未知名、必須入力不足、Flow/StateFlow種別の
不一致はGraph構築時またはcompile時の明示エラーとする。primary inputは一つだけであり、receiverを
どこへbindするかをシグネチャ推測に依存させない。

固定係数やFFT長を通常のFlow入力へ偽装しない。これらはConfigを使う。時変係数や制御値はFlow、
最新状態はStateFlow、Operation内部履歴はKernelStateへ置く。

### 3.2 統一Python実行契約

宣言OperationのPython実装は次の形へ統一する。

```python
def func(inputs, config):
    ...
```

- `inputs`は名前付き、不変、read-onlyな論理入力集合
- `config`はOperationが選択したscopeだけを見せる不変`ConfigView`
- `inputs`の値は一回の論理起動に対応し、Executorの物理batch件数ではない
- runtime context、Node、Port、Kernel ABI、Sessionを利用者関数の引数へ渡さない

単一outputは通常値、`cw.skip()`、`cw.emit_many()`を返せる。通常のlist/tupleは従来どおり一値で
あり、暗黙展開しない。複数Port出力は`OperationOutputSpec`の名前と一致する固定出力集合として扱う。
具体的なPython戻り値helper名は実装段階で決めるが、Port数と一Port内のEmission件数は混同しない。

### 3.3 `@cw.operation`

Flow境界となる本格的なDSP処理にはChronowire提供の`@cw.operation`を付ける。シミュレーション作者が
そのままDSP package作者を兼ねる場合も同じである。Operation内部だけで使う補助関数には不要である。

最小宣言は、入力と出力の値schemaが同じ、時間を保持、stateless、常に一件出力とする。

```python
@cw.operation(
    operation_id="example.normalize.v1",
    output="same",
)
def normalize(inputs, config):
    del config
    x = inputs["input"]
    return x / max(abs(x))
```

DSP処理は必要に応じて宣言を増やす。

```python
@cw.operation(
    operation_id="acme.dsp.fir.v1",
    inputs={
        "signal": cw.OperationInputSpec(
            primary=True,
            mode="synchronous",
            value=cw.ValueSpec(dtype="float64", shape=("samples", "channels")),
        ),
        "coefficients": cw.OperationInputSpec(
            mode="latest",
            value=cw.ValueSpec(dtype="float64", shape=(None,)),
        ),
    },
    output=cw.OperationOutputSpec(
        value=cw.ValueSpec(dtype="float64", shape=("samples", "channels")),
        time="preserve",
        emissions="one",
    ),
    config=cw.ConfigSpec(
        scope="dsp.fir",
        fields={"block_size": int, "channels": int},
    ),
    state="session",
    gap="reset",
    accepts_invalid=False,
)
def fir(inputs, config):
    return python_fir(
        inputs["signal"],
        inputs["coefficients"],
        block_size=config.block_size,
    )
```

`OperationSpec`が宣言すべき意味論は次である。

- 入力名、primary/synchronous/latest、required
- 入出力ValueSpecとshape解決規則
- time preserve、明示transform、出力interval規則
- 0/1/複数Emissionと一起動あたりの`max_items`
- stateless/session state、gap時のreset/accept
- DEGRADED/INVALIDの受理・伝播・fallback規則
- Config scopeと必須leaf

Python decoratorを付けただけでCython/C++実装になるわけではない。decoratorは言語非依存意味論と、
存在する場合のPython参照実装を登録する。概念上はOperationSpecの生成とPython Implementationの登録を
同時に行う糖衣構文であり、Python functionをOperationSpec fieldへ保存するものではない。

### 3.4 宣言のみOperation

Python実装を持たないOperationを正式に許容する。

```python
beamform = cw.declare_operation(
    operation_id="acme.dsp.beamform.v1",
    inputs={
        "spectrum": cw.OperationInputSpec(
            primary=True,
            value=cw.ValueSpec(
                dtype="complex128",
                shape=("frequency", "channels"),
            ),
        ),
    },
    output=cw.OperationOutputSpec(
        value=cw.ValueSpec(dtype="complex128", shape=("$config.beams", "frequency")),
        time="preserve",
    ),
    config=cw.ConfigSpec(
        scope="dsp.beamform",
        fields={"beams": int, "channels": int},
    ),
)
```

このOperationを`backend="python"`でcompileし、Python `ImplementationSpec`が登録されていなければ、
Node ID、output Port ID、operation ID、backend名を含む`MissingImplementation` compile errorにする。
宣言のみOperationは不完全なPython APIではなく、C++ first開発の正式な入口である。

### 3.5 plain callable

decoratorのないplain callableは試作用Python-only Operationとして残す。

```python
preview = signal.map(lambda x: x * 0.5)
```

互換adapterが次のdefaultを与える。

- representation: `python_opaque`
- state: stateless
- time: preserve
- output Port: 1
- Emission: 1件。`skip`/`emit_many`を許す場合は既存明示adapterを使用
- implementation: Python only

plain callableはPython Executorの試作経路には十分だが、fixed shapeの静的証明、portable native
実行、Cython/C++実装交換には使わない。それらが必要になった時点で`@cw.operation`または
`declare_operation()`へ移行する。v0.4互換期間中は既存callable signatureと`config_paths`を
adapterが受理するが、新しいOperationの統一契約には含めない。

## 4. Config設計

### 4.1 scope選択

Operationは`config_scope="dsp.fir"`または`ConfigSpec(scope="dsp.fir", ...)`で、現在のFlowが参照する
Configからsubtreeを選ぶ。実行関数へはそのsubtreeだけの`ConfigView`を渡す。

```python
config = cw.Config(
    system={"fs": 48000},
    dsp={
        "fft": {"nfft": 1024},
        "fir": {"block_size": 256, "taps": [0.25, 0.5, 0.25]},
    },
)
```

`dsp.fir` Operationから`dsp.fft.nfft`やroot全体を暗黙に参照できない。複数種類の固定設定を渡すために
`FirOptions`等の別経路を作らず、Config subtree内へ整理する。scopeをまたぐ本当に独立した依存が必要な
場合は、複数scopeを許す前にConfig構造を見直す。v1設計では一Operation一scopeを基本とする。

### 4.2 leaf依存追跡

`ConfigSpec.fields`は選択scope内の相対leaf pathと型を宣言する。compileは次を記録する。

- Config scope IDとresolved digest
- Operationが必要とする相対leaf path
- 各leafの型検証結果と解決元scope
- secret値を除く再現用値、またはredacted digest

現行`flow.map(func, config_paths=(...))`はNode呼出し側が依存を宣言する。新設計ではDSP package作者が
`OperationSpec.ConfigSpec`へ依存を置き、すべての利用者が同じ契約を共有する。plain callableだけは
互換のため`config_paths`を残し、省略時はscope全体依存としてcompile cacheを保守的に扱う。

固定係数、FFT長、channel数はConfigである。時変係数、steering更新、制御値、latest calibrationは
`inputs` Edgeである。FIR履歴、積分器、FFT workspace、native handleはKernelStateである。

## 5. ValueSpecとshape証明

### 5.1 item schemaと物理batchの分離

PortのValueSpecはEmission一件の値schemaを表す。Executorが複数Emissionをまとめる物理batch件数を
item shapeへ追加してはならない。

```text
Port item shape:       (frequency, channels)
Executor physical batch: N items × (frequency, channels)
```

ValueSpecは少なくとも次を持つ。

- `dtype`
- `shape`
- `device`
- `representation`
- `read_only`

stride、alignment、contiguous条件はportableなValueSpecの要求または選択Implementationの制約として
扱い、runtime BufferViewで再検証する。

### 5.2 dimension表現

compile前ValueSpecの最小dimension表現は次とする。

| 表現 | 例 | 意味 |
|---|---|---|
| 固定値 | `1024` | 常に同じ正のdimension |
| symbolic | `"channels"` | 同一Operation内の同名symbolは一致必須 |
| Config由来 | `"$config.nfft"` | 選択ConfigViewのleafで解決 |
| 任意 | `None` | 値はruntimeまで未固定 |

予約prefix`$config.`とsymbol helperの最終的なPython表記は実装前にAPI reviewするが、意味論とIRへの
正規化は上表で固定する。同名symbolは入力間、入力と出力、Config制約のすべてでunifyする。

### 5.3 伝播規則

Compilerは次の順でValueSpecを伝播する。

```text
SOURCE  宣言された一件のValueSpecを生成
RATE    item shapeを変更しない
FRAME   先頭へ固定frame size dimensionを追加
MAP     OperationSpecで入力をunifyし、出力ValueSpecを解決
```

FFT、共分散、beamforming、resampling等のshape規則はBackend実装へ置かない。Python/C++のどちらを
選んでも同じOperationSpecから同じresolved output schemaを得る必要がある。shape resolverはcompile
時だけ実行し、PortablePlanIRへcallableを入れず、解決済みschemaだけを保存する。

固定値、symbol、Config dimensionだけで書けない`rfft_bins(nfft)`等は、Operation declaration側の
compile-time resolverまたは将来の最小shape式で表す。resolverはBackend implementationではなく
OperationSpecの一部として全Backendに一度だけ適用する。Python callableを登録形式に採用してもPlanへ
serializeせず、IRには入力schema、Config digest、resolved出力schemaだけを保存する。

`python_opaque`は許容する。ただしOperationまたは選択Implementationがfixed shapeを必須とし、
compileが証明できなければcompile errorにする。runtime ABI境界では、resolved schemaに対してbyte長、
stride、alignment、device、read-onlyを再検証する。

shape errorには少なくとも次を含める。

- Node IDとPort ID
- operation ID
- input名
- expected schemaとactual schema
- 違反したdimension index、symbol名、Config leaf

## 6. OperationSpecとImplementationの分離

### 6.1 OperationSpec

OperationSpecはPython、Cython、C++から独立した意味論の正本である。DSP packageが所有し、同じ
operation IDの全Implementationが共有する。実装言語、SIMD種別、module pathを含めない。

operation IDはpackage内で安定かつversion付きとする。入出力、時間、status、stateの互換性を壊す変更は
新しいoperation ID versionを使う。

### 6.2 ImplementationSpec

ImplementationSpecは同じoperation IDを実行する一つの候補を表す。

- implementation IDとversion
- target language/runtime: Python、Cython、C ABI
- ABI version
- 対応するresolved dtype、rank、shape制約
- workspace size/alignment/device
- create/process/flush/destroy capability
- determinism、thread safety、in-place、required CPU feature
- binding slotとmodule ABI

Python decoratorが持つfunctionはPython Implementationのprocess-local登録に使えるが、IRへ保存しない。
Cython/C++ implementationはnative module manifestがImplementationSpecを提供する。

### 6.3 compile後のPlan

PortablePlanIRのOperationDescriptorには次を保存する。

- operation ID
- Node ID、input/output Port IDと入力名
- resolved input/output ValueSchemaDescriptor
- Config scope ID、leaf dependency、digest
- time、Emission、status、state、gap規則
- 選択implementation ID、ABI version、execution domain
- binding slot、workspace descriptor

次は保存しない。

- Python function、decorator object
- shape resolver callable
- native pointer、module handle、allocator
- shared library path
- process-local objectまたはPython class名

これらはExecutionBindings内のImplementationBindingに置く。別processでPortablePlanIRを読む場合だけ、
operation ID、implementation ID、ABI version、binding slotを使ってmoduleを再bindする。同一processの
compile直後にBackendとExecutorを利用者へ二重指定させない。

## 7. BackendとExecutor

### 7.1 Backendは実装を選ぶ

Implementation選択は明示的にする。`backend`は互換性のため残すPlan既定selectorであり、
`implementations`はoperation IDごとのoverrideである。同じoperation IDを使う全Nodeには同じselectorを
適用する。Executorはこの選択に関与しない。

```python
python_plan = cw.compile(outputs, backend="python")
mixed_plan = cw.compile(
    outputs,
    backend="python",
    implementations={"dsp.beamformer.v1": native_backend},
)
native_plan = cw.compile(outputs, backend=native_backend)
```

`backend="python"`は未指定OperationのPython Implementationを選ぶ。`NativeModuleBackend`を既定または
overrideに指定したOperationは、decoratorに付属するPython計算本体を実行しない。ただしOperationSpecは
すべてのImplementationに共通する意味論の正本として使う。`PortablePlanIR.backend`は単一domainならその
名前、複数domainなら`mixed`とし、Nodeごとの選択結果はImplementationDescriptorへ記録する。

選択BackendにImplementationがなければ既定でcompile errorにする。黙ってPythonへfallbackしない。
将来`PreferNative`等のpolicyを追加する場合は、選べなかったimplementation、理由、選択したfallbackを
Plan DiagnosticとOperationDescriptorへ必ず記録する。

C++ Backendは同じImplementation内でscalar、AVX2、NEON等を自動選択してよい。選択したvariant、CPU
feature、implementation IDはPlanへ記録し、再現性と性能測定で参照可能にする。

### 7.2 ExecutorはPlanを運用する

BackendはOperation implementationの選択とcompileだけを行う。Executorはcompile後の次を所有する。

- Source requestとready判定
- RATE、FRAME、同期、latest入力
- Edge buffer、fan-out寿命、collector
- KernelStateのcreate/process/flush/destroy
- gap reset、status/Diagnostic配送
- ExtensionとPython Stage境界

組合せの意味は次である。

| Backend | Executor | 用途 |
|---|---|---|
| Python | PythonExecutor | 実行可能な参照実装、試作、デバッグ |
| C++ | PythonExecutor | native Operation単体のconformance |
| C++ | CppExecutor | Schedulerを含むcompile後Plan全体のnative運用 |
| Pythonのみ | CppExecutor | 単一の最大Python islandを協調的にyield/resumeする |
| Python/native mixed | CppExecutor | Python island境界でbatchをyield/resumeするhybrid運用 |

Backendを`cpp`にしただけでSchedulerがC++になるわけではない。Executorを`cpp`にしたとき、Planが要求する
ImplementationBindingはcompile時に選択済みであり、利用者へ再度Backendを指定させない。

PythonExecutorは最終的な高速実行の必須構成要素ではないが、実行可能な参照実装、plain callable、デバッグ、
Operation単位のPython/C ABI同値性試験を担うため残す。C ABI Implementationを選択したPlanでも
PythonExecutorはそのfunction tableをEmission単位で直接呼び、Python Implementationへfallbackしない。
CppExecutorは同じ選択済みbindingをC++ runtimeから直接呼び、完全native Planではhot pathに
Python dispatchを入れない。

Implementation言語とExecutorは直交する。したがってすべてのOperation本体がPython関数でも
CppExecutorで実行可能とする。ただし性能分類は別であり、all-nativeは
`python_free_hot_path=True`、mixedは`hybrid`、all-Pythonは`python_stage_dominated`と記録する。
実行可能性を`native_compatible`だけで判定せず、各Stageにnative runner、Python Stage runner、
または対応済み境界codecのいずれかがあれば受理する。

CppExecutorのC++ runtimeはPython C APIやPython callbackを直接呼ばない。`advance()`は
`Completed`または`NeedsPython(stage_id, input_batches)`を返し、Cython/Python adapterがGILを
取得してrun-local `PythonStageSession`をbatch実行する。adapterはshape、Emission件数、
interval、sequence、status、Diagnosticを検証し、`resume(stage_id, output_batches)`でC++
Schedulerを継続する。連続するPython OperationはCompilerが最大Python islandへまとめ、
Emission単位で往復しない。

`python_opaque`はPython island内だけで許容し、native Stageへ出るPortは明示ValueSchemaと
境界codecを必須とする。固定dtype/shape境界はC++所有bufferをread-only memoryviewとして
貸し、buffer protocol適合出力はborrow/zero-copyを優先する。不連続または契約不適合の
場合だけ境界で1回copyし、fan-outはread-only batchを共有する。

2026-07-22時点では、all-Python one-shot Plan、単一Python island往復、およびnative Source側から
始まる線形複数island Planを実装済みである。StageDescriptorは外部入力・出力Port IDを保持し、mixed境界は
C++結果vectorはread-only Python buffer ownerへmoveする。通常のPython実装へはtupleへ一回copyするが、
`accepts_readonly_buffers=True`を持つImplementationへは各itemをflatなread-only memoryviewで貸し、
同じowner上の連続viewを返した場合は合成native ingressもborrowする。この能力はcanonical Python値を
拒否する指定ではなく追加受理能力であり、同じImplementationはPythonExecutorの通常値も扱う。
0/1/複数Emission、RATE/FRAME、fan-out、status/Diagnostic、例外後の再実行をPythonExecutorと
照合済みである。native→Python複数入力は`synchronous`完全interval一致と`latest`選択を
単一・複数islandで実装済みである。fixed-schemaの明示opt-in borrowと単一copy fallbackも実装済みである。
Python→native複数ingress、Pythonが新規生成したbufferのborrow、
mixed Cpp Sessionは次段階であり、未対応時はNode/Port/Stage/bindingを含む明示エラーにする。

意味論の正本はどちらのExecutor実装でもない。`OperationSpec`、`PortablePlanIR`、および
論理時間、Emission件数、status、Diagnostic、buffer、lifecycleの設計契約を正本とする。
PythonExecutorとCppExecutorはこの契約に対する独立実装であり、契約から固定したgolden trace、
property test、Executor間相互比較の三つで検証する。一方の実行結果を無条件に他方のoracleとしない。

### 7.3 Chronowire本体を改修する境界

新しいDSP Operationや、既存ABI上のPython/Cython/C++ Implementationを追加するだけなら、
Chronowire本体の`PythonBackend`、`NativeModuleBackend`、`PythonExecutor`、`CppExecutor`は
原則として改修しない。DSP package側の`OperationSpec`、Implementation、native module manifest
entry、契約conformance testだけで完結させる。

BackendまたはExecutor本体を改修するのは、次のように「実装を載せる器」の契約を
拡張する場合に限る。

- 新dtype、可変shape、複数output Port、0件または複数Emission
- Config packing、workspace、flush、checkpoint、新しいlifecycle
- 新ABI/process model、Implementation選択policy
- Scheduler、buffer、論理時間、status/Diagnostic配送などPlanの実行意味論

最後の項目を拡張する場合はPythonExecutorとCppExecutorの両方に実装が必要になり得るが、
それはアルゴリズム追加の二重実装ではなく、独立Executorが同じPlan契約を実行するための
必要経費である。全Executorの同時対応は必須とせず、未対応capabilityはNode、Port、
binding slot、違反契約を含む明示エラーで拒否し、暗黙fallbackしない。

## 8. Native moduleとC ABI

C++/Cython sourceへPython decoratorは書かない。native moduleはversion付きC ABI module tableを
exportする。

Chronowireの契約を課す境界はOperation wrapperだけとする。既存DSP libraryや内部の数値アルゴリズムに
`Flow`、`Node`、`Port`、`Emission`、`PortablePlanIR`を理解させない。wrapperは次の責務を完全に引き受ける。

- operation/implementation IDとABI versionの公開
- resolved input/output schema、byte長、stride、alignment、read-onlyの検証
- ConfigViewからDSP library固有parameterへのimmutable packing
- `create/process/flush/destroy`とrun-local state/workspaceのライフサイクル変換
- DSPの結果、劣化、数値失敗、例外をstatus、Diagnostic、ABI errorへ変換
- input/output bufferのownershipと寿命をChronowire runtimeとDSP libraryの間で調停

wrapperの背後は通常のPython、Cython、C++ library APIでよく、Chronowireと独立にunit testできる。
Cython wrapperはtyped memoryviewまたはnative pointerに変換し、内側のhot loopを`nogil`のC/C++コードに
委譲してよい。C++ wrapperも同様に、既存classやfunctionをC ABIへ適配する。これによりDSP
packageはChronowire wrapperだけを追加し、アルゴリズム本体の責務と依存を増やさない。

「wrapperだけに契約を課す」とは契約を緩めることではない。wrapperはOperationSpecとABIの全条件を
守り、内部DSPが返したshape、status、所有権が不適合なら、Node、Port、operation ID、違反契約を
含むcompile、bindまたはruntime境界エラーにする。アルゴリズム内部で生成された安全な劣化結果は
`DEGRADED`とDiagnosticに変換し、例外で失わない。

v0.4の正本headerは`src/chronowire/native_operation_abi.h`とする。moduleは
`chronowire_operation_module_v1()` symbolから`CwOperationModuleV1`を返す。公開loaderは
`NativeOperationModule(path)`、Backendは`NativeModuleBackend(module)`である。library path、CDLL handle、
function addressは`NativeOperationEntry`と`NativeOperationRuntimeBinding`だけがprocess-localに保持する。

```c
typedef struct {
    const char* operation_id;
    const char* implementation_id;
    const char* abi_version;
    cw_create_fn create;
    cw_process_fn process;
    cw_flush_fn flush;
    cw_destroy_fn destroy;
} CwOperationEntryV1;

typedef struct {
    const char* module_abi_version;
    size_t operation_count;
    const CwOperationEntryV1* operations;
} CwOperationModuleV1;
```

ABI関数はC++例外を境界外へ出さず、error codeとDiagnostic writerを使う。createはresolved schema、
Config digest/定数binding、workspaceを受けてrun-local sessionを作る。processは名前から固定されたinput
slot順のread-only BufferView列を受け、出力BufferView列とEmission metadataを返す。flushは0件以上の
残留出力を返し、destroyは部分初期化を含めて一度だけ安全に解放する。

native implementationはsymbolic shape規則を重複実装しない。Compilerが解決したschemaを受理できるか、
runtime bufferがそのschema、byte長、stride、alignment、deviceに一致するかだけを検証する。module handle
とfunction table pointerはImplementationBindingがprocess-localに保持する。

v0.4 module ABIは成立経路を固定するため、連続float64の固定shape input、単一固定shape output、process
一回につき一Emissionへ限定する。ConfigはOperationSpecのfield順にfloat64 scalarまたは数値tupleへflatten
してcreateへ渡す。processは0から2のstatusと、任意のDiagnostic severity/code/messageを返せる。Executorは
入力Diagnosticを保持し、module DiagnosticへNode、Port、intervalを付与する。create/process/destroyは必須、
flush pointerとcapability flagは一致必須である。v0.4 CppExecutorはflush出力を要求するOperationをまだ受理
しない。可変出力、複数output Port、typed Config table、device BufferViewはABI versionを更新して追加する。

## 9. 状態、時間、status、劣化

### 9.1 状態

- `stateless`: KernelStateは軽量でもよく、run間で状態を共有しない
- `session`: `Session.start()`からclose/cancelまで履歴とworkspaceを保持
- gap=`reset`: 欠落後の最初の入力前に新しい状態へreset
- gap=`accept`: OperationSpecが欠落を受理しDiagnosticを出力へ伝播するときだけ許可

PlanやImplementationBindingへrun-local可変状態を置かない。例外後のKernelStateを再利用せず、
同じPlanから新しいsessionを作れる状態へ戻す。

### 9.2 時間とEmission件数

単純規則はOperationSpec fieldで表す。

- `time="preserve"`: primary input intervalを保持
- `time="explicit"`: Operationが新しいintervalを返す境界
- `emissions="one"`: 常に一件
- `emissions="zero_or_one"`: skip可能
- `emissions="many", max_items=N`: 一起動で最大N件

明示resamplingは`time="explicit"` Operationとして旧格子を終了し、その後に`rate().frame()`で新格子を
確定する。Backendが独自に時間規則を変えない。

### 9.3 statusとDiagnostic

DEGRADEDは既存Diagnosticを保持して通常実行する。INVALIDを受理しないOperationはprocessを呼ばず、
`INVALID_INPUT_PROPAGATED`を付けて伝播する。受理、fallback、status改善を行う場合はOperationSpecへ宣言し、
理由のDiagnosticを追加する。安全なfallback、不十分な積分、観測可能な数値失敗を例外で失わない。

## 10. 二つの開発経路

### 10.1 C++ first

```text
declare_operation
    ↓ OperationSpecをreview、shape/time/statusをcompileで検証
C++ native moduleが同じoperation IDのImplementationSpecを提供
    ↓
C++ Backendでcompile
    ↓
PythonExecutorでOperation conformance
    ↓
CppExecutorでPlan全体をnative運用
```

```python
def resolve_rfft_shape(input_specs, config):
    return (config.nfft // 2 + 1, input_specs["samples"].shape[1])


fft = cw.declare_operation(
    operation_id="acme.dsp.fft.v1",
    inputs={
        "samples": cw.OperationInputSpec(
            primary=True,
            value=cw.ValueSpec(dtype="float64", shape=("$config.nfft", "channels")),
        )
    },
    output=cw.OperationOutputSpec(
        value=cw.ValueSpec(dtype="complex128", shape=(None, "channels")),
        time="preserve",
    ),
    config=cw.ConfigSpec(scope="dsp.fft", fields={"nfft": int}),
    shape_resolver=resolve_rfft_shape,
)

spectrum = frames.map(fft)
fft_module = cw.NativeOperationModule("/path/to/libacme_dsp.so")
fft_backend = cw.NativeModuleBackend(fft_module)
plan = cw.compile([cw.output(spectrum)], backend=fft_backend)
```

Python実装がなくてもC++ Backendでは成立する。Python Backendを選んだ場合だけ明示的な
MissingImplementationとなる。

### 10.2 Python first

```text
@operationでOperationSpecとPython参照実装を定義
    ↓ Python Backend + PythonExecutorで検証
同じoperation IDのC++ Implementationを追加
    ↓ C++ Backend + PythonExecutorでconformance
C++ Backend + CppExecutorへ変更
```

```python
@cw.operation(
    operation_id="acme.dsp.power.v1",
    inputs={
        "spectrum": cw.OperationInputSpec(
            primary=True,
            value=cw.ValueSpec(dtype="complex128", shape=("frequency", "channels")),
        )
    },
    output=cw.OperationOutputSpec(
        value=cw.ValueSpec(dtype="float64", shape=("frequency", "channels")),
        time="preserve",
    ),
)
def power(inputs, config):
    del config
    return abs(inputs["spectrum"]) ** 2

power_flow = spectrum.map(power)
python_plan = cw.compile([cw.output(power_flow)], backend="python")
cpp_plan = cw.compile([cw.output(power_flow)], backend=power_native_backend)
```

二経路ともFlow記述とOperationSpecを変更しない。変えるのはImplementation登録とBackend選択だけである。

## 11. 具体的な契約違反

### 11.1 複数入力とStateFlow

```python
steered = spectrum.map(
    beamform,
    steering=steering_flow,
    calibration=calibration_state,
)
```

`steering`がsynchronous、`calibration`がlatestと宣言されている場合、Flow/StateFlowを逆にbindすると
compile errorにする。同期入力は既存のduration/period/offset証明を適用し、latestはprimary interval
start以前の最新確定値を使う。

### 11.2 shape不一致

`beamform`が`channels` symbolを両入力で共有し、spectrumが8 channel、steeringが6 channelなら、
Backend選択前にcompile errorにする。

```text
ShapeMismatch:
  node=12 port=19 operation=acme.dsp.beamform.v1
  input=steering dimension=1 symbol=channels
  expected=8 actual=6
```

Python Backendだけ偶然broadcastできても受理しない。OperationSpecの意味論をBackendごとに変えない。

### 11.3 implementation不足

```text
MissingImplementation:
  node=12 port=20 operation=acme.dsp.beamform.v1
  backend=cpp required_abi=chronowire.operation.c.v1
```

Python参照実装が存在しても`NativeModuleBackend`から暗黙に呼ばない。

## 12. v0.4からの移行

### 12.1 名称対応

| v0.4実装 | Operation設計 |
|---|---|
| v0.4 `Kernel` protocol | 内部`KernelProvider`。Operation移行用legacy入口 |
| 開発中の`CompiledKernel` | `Kernel`。compile済み不変factory |
| 開発中の`CompiledKernelSession` | `KernelState`。run-local実行主体 |
| `NativeValueSchemaProvider` | OperationSpecのcompile-time shape規則 |
| `NativeKernelRuntimeBinding` | `ImplementationBinding` |
| `KernelAbiDescriptor` | OperationDescriptorが参照するImplementation/ABI descriptor |

正式公開前のため、Pythonの旧class名と旧factory methodは削除し、deprecated aliasを設けない。
PortablePlanIR schemaとC ABI IDはPython class名を保存しないため変更しない。

互換性は次の境界ごとに扱う。

| 境界 | v0.4方針 |
|---|---|
| 公開Python名 | `Plan`、`Session`、`SessionState`、`Kernel`、`KernelState`、`BoundPlan`だけを正式名として公開する |
| 内部Python名 | Executor最適化用runnerは内部名とし、公開Session型を分けない |
| method | Session生成は`create_session()`だけとし、一括・段階実行を同じSessionから選ぶ |
| PortablePlanIR | `kind="execution_plan"`等の既存serialized IDはreader互換性のため変更しない。Python class名を追加保存しない |
| C ABI | module ABI、operation ID、implementation ID、binding slotは変更しない |
| import path | 開発中の旧名は削除し、移行入口を設けない。内部module pathの互換性は保証しない |
| pickle | Plan、Session、KernelStateの永続化契約には採用しない。process間移送はPortablePlanIRと明示bindingを使う |

`Plan`の不変性は、compile済みNode列、出力契約、Kernel mapping、Backend選択、PortablePlanIRを
run中に変更しないことを意味する。SourceやExtension等のprocess-local実体とKernelStateはSessionが所有し、
同じPlanから作る別Sessionへ共有しない。

### 12.2 固定CBF Operation

固定CBF参照packageは、Flow利用者へKernel factoryを見せずOperation helperを公開する。

```python
from chronowire_reference import CythonCbfBackend, fixed_cbf, fixed_cbf_operation

beam = fixed_cbf(frames, weights)
plan = cw.compile(
    [cw.output(beam, collector=cw.Latest())],
    implementations={fixed_cbf_operation.operation_id: CythonCbfBackend()},
)
```

`fixed_cbf()`は係数を不変な`cbf` Config scopeへ記録し、`fixed_cbf_operation`をFlowへ追加する。
OperationSpecはshape、time、status、Config leafを宣言し、Python/Cython/C++実装は同じoperation IDへ
登録する。利用者はKernel class、ABI ID、session factoryを選ばない。固定shapeを証明できない
`python_opaque`入力は、native境界を推測せずcompile errorにする。

### 12.3 互換方針

- plain callableはPython-only adapterとして維持する
- `flow.map(func, named_flow=...)`とStateFlowのGraph意味論を維持する
- Configの不変scope、属性アクセス、leaf mergeを維持する
- plain callableの`config_paths`は移行期間中維持する
- status、Diagnostic、0/1/複数Emission、gap規則をOperationDescriptorへ移す
- legacy Kernel objectは内部実行契約として残すが、新規DSP公開APIはOperationSpecを正本にする
- schema 0.3の`KernelAbiDescriptor` readerを維持し、新schema exportはOperationDescriptorを正規形にする
- CppExecutorのidentity/固定CBF/MVDR resolver tableはOperation固有知識をExecutorに持つ
  過渡的実装であり、外部C ABI module tableの汎用dynamic bindingへ置換する。
  Operation追加ごとにresolver tableやExecutorを改修する構造は継続しない

## 13. 段階的実装順序

設計確定後は次の順序で実装する。2026-07-22時点の状態も併記する。

1. **実装済み**: `OperationSpec`、input/output、ValueSpec、ConfigSpecのimmutable modelとvalidation
2. **未実装**: plain callableとlegacy KernelをOperationへ正規化するadapter
3. **初期実装済み**: SOURCE→RATE→FRAME→MAPのcompile-time shape unification。Backend選択前に
   固定値、symbol、Config dimensionを検証する
4. **初期実装済み**: schema 0.4のOperationDescriptor、ImplementationDescriptor、binding slotと、
   schema 0.1〜0.3互換reader。resolved schemaとConfig依存を保存する
5. **初期実装済み**: Python Backendとrunごとに生成するPython KernelState。公開のmutable state
   factoryは未実装
6. **実装済み**: MissingImplementation、shape、Config scope/leaf、Python/C ABI ImplementationBindingの
   ID/ABI不一致を明示エラーにする
7. **初期実装済み**: process-local ImplementationBindingと別process相当の再bind。C ABI v1 module table、
   明示path loader、module lifetime管理を実装。package discoveryと署名検証は未実装
8. **参照経路を実装済み**: operation IDごとの`implementations` overrideと、C++ Backend選択結果を
   PythonExecutorで実行するC ABI conformance。周期MVDRの累積共分散、重み生成、latest適用を
   同一OperationSpecで検証する。汎用module loaderは手順7の残件
9. **初期実装済み**: schema 0.4 OperationDescriptorをCppExecutorが読み、複数入力、LATEST、SAMPLE、
   run-local stateを登録済みABIまたは外部C ABI module tableから実行する。固定shape float64単一outputが範囲
10. **実装済み**: Fixed CBF参照packageを`fixed_cbf()`／`fixed_cbf_operation`へ移行し、開発中の
    `FixedCbfKernel`公開名は互換aliasを残さず削除
11. **初期実装済み**: `native_operation_include_dir()`でwrapper向けABI headerを公開し、
   DSP本体をChronowireに依存させない外部module build境界を固定
12. **線形複数Python islandまで初期実装済み**: CppRuntimeMetricsでGIL解放契約、native Stage dispatch、
   Python境界callback、公開Emission復元、batch変換を分離計数。native prefix/plain callable/
   native区間間を一回copyで往復できる。native→Pythonのsynchronous/latest複数入力境界も
   実装済み。明示opt-inした固定shape単一ingressはzero-copyで往復できる。Python生成bufferの
   zero-copyとPython→native複数ingressは未実装

初期実装では`@cw.operation`と`cw.declare_operation()`を`Flow.map()`へ渡せる。receiver、同期Flow、
latest StateFlowを宣言名へbindし、Python実装へ不変な`inputs` mappingと選択済み`ConfigView`だけを渡す。
複数Port、DEGRADED/INVALID伝播、宣言のみOperationの明示的な実装不足も既存runtime上で検証する。

宣言Operationを含むPlanはPortablePlanIR schema 0.4をexportする。`OperationDescriptor`はNode/Port、
resolved schema、Config scope/path/digest/leaf型、time/Emission/status/state/gap規則、選択implementationと
binding slotを正本として記録する。`ImplementationDescriptor`はID、Backend、ABI、process model、
workspace、CPU featureを記録し、Python callable、shape resolver、native pointer、module handle、library
pathは保存しない。legacy KernelだけのPlanはschema 0.3を維持し、0.1〜0.3 readerも削除しない。

schema 0.4のOperation Planは、IR、Config、Source/Collector、およびprocess-local
`ImplementationBinding`から再bindして実行できる。native moduleは明示pathからloadし、operation ID、
implementation ID、ABI versionを照合する。暗黙のfilesystem探索は行わない。

各段階で設計契約から固定したgolden traceに対する値、interval、sequence、status、
Diagnostic、metadata同値を確認する。PythonExecutorのtraceは参照結果の一つであり、
契約そのものの代わりにしない。

### 13.1 周期MVDRの受入構造

参照packageは次のFlowを構築する。

```text
Source → RATE → FRAME ───────────────────────────────→ apply_weights → beam
                   └→ covariance(session state) → SAMPLE → weights ─latest─┘
```

`SAMPLE`はframeのrate変換ではなく、入力intervalを保ったまま更新境界の完成Emissionだけを選ぶ。
`period`がframe periodの整数倍であることをcompile時に証明し、端数frameや未使用frameを許さない。
共分散積分はKernelState、最新重みはLATEST Edge、固定steeringとdiagonal loadingはConfigに置く。
積分不足時も安全に解ける対角loading済み共分散を`DEGRADED`として出力し、重みとbeamまでDiagnosticを
保持する。実装は実数の小規模conformance用であり、複素FFT bin別の本番MVDRはDSP packageが同じ
OperationSpec/ImplementationSpec境界で提供する。

## 14. 破壊的変更候補と未決事項

### 14.1 破壊的変更候補

- DSP packageが実装objectをFlowへ渡す旧APIの削除（固定CBF参照packageでは実施済み）
- 宣言Operationでの`config_paths` map引数廃止とConfigSpecへの集約
- named通常値をNode parameterとして渡す経路の縮小。固定値はConfig、時変値はFlowへ移す
- PortablePlanIRのKernel中心descriptorからOperation/Implementation中心descriptorへのschema更新

0.xの間にmigration warningと変換例を提供し、v1.0前に公開面を整理する。

### 14.2 実装前に決める項目

1. `ValueSpec.shape`のsymbol/Config dimensionを表す最終Python構文。意味論は本書どおり固定する。
2. 複数output Portを返す公開helper名。Port数とEmission件数を分離することは確定済み。
3. Operation registryの発見範囲と同じoperation/implementation ID重複時の明示的解決API。
4. native moduleの配布・署名、および明示pathより上位のpackage discovery規則。
5. Config値をImplementationBindingへimmutable bytesとして渡す標準型の範囲。
6. optional inputとdefault値をConfig、欠損同期入力、OperationSpecのどこで表すか。
7. `time="explicit"`のportable出力interval descriptorとshape resolver DSLの最小表現。
8. legacy Kernel adapterを削除するversionとPortablePlanIR schema migration期間。

未決事項をBackend側の暗黙推測やPython fallbackで埋めない。採用前はcompile errorまたは
`python_opaque`境界として明示する。

## 15. 他設計書との関係

- 公開Flow API: [02_公開API設計.md](02_公開API設計.md)
- Backend、Config、v0.4 Kernel移行: [05_Kernel_Backend_Config設計.md](05_Kernel_Backend_Config設計.md)
- compile、session、buffer planning: [04_compileと実行時設計.md](04_compileと実行時設計.md)
- Native Backend/Executor責務: [10_Native_Executor設計方針.md](10_Native_Executor設計方針.md)
- PortablePlanIR、buffer、shape descriptor: [11_Buffer_Scheduler_PortablePlanIR設計.md](11_Buffer_Scheduler_PortablePlanIR設計.md)
- status、Diagnostic、Extension: [06_Extension_export_診断設計.md](06_Extension_export_診断設計.md)
