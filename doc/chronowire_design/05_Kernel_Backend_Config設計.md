# Operation移行・Backend・Config設計

> **移行上の位置付け:** 本書の`Kernel`、`CompiledKernel`、`CompiledKernelSession`はv0.4現実装を
> 説明する旧名称である。新しい公開API、言語非依存OperationSpec、Implementation、shape、C ABIの
> 正本は[14_Operation設計.md](14_Operation設計.md)とする。既存Kernel APIは段階的adapterとしてのみ
> 維持し、Flow利用者へABIやSessionを露出しない。

# 1. Kernel

Chronowireは処理アルゴリズムを知らない。Kernel protocolを通じてPython関数、Pythonクラス、Cython、C++を同一Graph上で扱う。

Operation移行後は「同じOperationSpecにPython/Cython/C++のImplementationSpecを登録する」と表現する。
以下のprotocolは新設計の内部`CompiledOperation`/`OperationSession`へ責務を引き継ぐ。

## 1.1 compile/run分離

```python
class Kernel(Protocol):
    def compile(
        self,
        context: CompileContext,
    ) -> CompiledKernel:
        ...
```

```python
class CompiledKernel(Protocol):
    def create_session(self) -> CompiledKernelSession:
        ...

class CompiledKernelSession(Protocol):
    def run(
        self,
        inputs: tuple[Any, ...],
        context: RunContext,
    ) -> KernelResult:
        ...
```

compile時に行う処理:

- Config解決
- shape/dtype検証
- 係数生成
- steering vector生成
- FFT plan生成
- workspace requirementの決定
- backend選択
- SIMD実装選択

run時は計算だけを行う。

`KernelResult`はoutput Portごとの0件以上のEmissionを返す。Port数とemit件数を混同しない。

```python
@dataclass(frozen=True)
class KernelResult:
    outputs: tuple[tuple[Emission[Any], ...], ...]
```

安全なfallbackを生成できる数値的不足は例外ではなく、`DEGRADED`または`INVALID`なEmissionとして返す。契約違反や安全に継続できない実装例外だけを`KernelExecutionError`にする。

v0.1のstatus規則:

- `OK`: 通常どおりKernelへ渡す
- `DEGRADED`: Diagnosticを保持したまま通常どおりKernelへ渡す
- `INVALID`: Kernelが`accepts_invalid=True`を宣言しない限りKernelを呼ばず、`INVALID`を後段へ伝播する

Kernelがstatusを変更する場合は、新しいDiagnosticで理由を追加する。既存の劣化・無効理由を削除してはならない。

## 1.2 Python callable

```python
def normalize(x):
    return x / np.max(np.abs(x))

out = flow.map(normalize)
```

内部で`PythonCallableKernel`へ変換する。

シグネチャ解析により`config`注入の有無をcompile時に確定する。

Python callableは実行時の特別分岐で直接呼ばず、`CompiledKernelSession`を生成する通常のKernel lifecycleへ正規化する。Python callableはstateless契約とし、run間で継続する可変状態が必要な処理は明示的なstateful Kernelとして実装する。

## 1.3 stateful Kernel

`CompiledKernel`はcompile済み係数、native handle、実行方式など、複数run間で共有可能な情報を保持するfactoryである。可変状態は`create_session()`が生成する`CompiledKernelSession`だけに保持する。

必要な契約:

- runごとに新しいsessionを生成
- thread safety
- snapshot可否
- run間の状態継続可否

v0.1の`ExecutionSession.run()`はrun開始時にsessionを生成し、前回runの状態を構造的に参照できないようにする。v0.2の`PlanSession`は`start()`時に一度だけ`CompiledKernelSession`を生成し、複数の`run_until()`間で同じinstanceを保持する。`close()`、`cancel()`、または実行失敗後にそのinstanceを再利用せず、同じExecutionPlanから作る次のPlanSessionは必ず新しいKernel sessionを得る。

## 1.4 複数出力Kernel

Graph内部は複数output Portを扱える設計にする。

公開Flow APIはv0.1で単一output Portに限定する。Graph IRとKernelResultは複数Portを表現できるが、公開APIから生成しない。

v0.2で公開する基本形:

```python
a, b = flow.map_outputs(split, output_count=2)
```

Kernelは`kernel_outputs(a, b)`を返し、`map_outputs()`は固定長の通常tupleとしてFlow handle列を返す。通常tupleは従来どおり一値であり暗黙展開しない。`FlowTuple`や`MultiFlow`という追加公開classは設けない。未観測sibling Portはruntime bufferへ保持せず、必要Portの実行を妨げない。

plain callableには`callable_kernel()`で`max_items`、`accepts_invalid`、`time_transform`、`GapPolicy`を一括宣言できる。`time_transform="preserve"`は入力intervalを維持し、`"explicit"`は外部resampling Kernelが出力intervalを決定する境界を表す。Graph構築後にこれらを上書きせず、PortablePlanIRへ固定する。

# 2. Backend

## 2.1 protocol

Operation移行後の概念protocolは次とする。

```python
class Backend(Protocol):
    name: str

    def compile_operation(
        self,
        operation: OperationSpec,
        implementation: ImplementationSpec,
        context: CompileContext,
    ) -> CompiledOperation:
        ...
```

Backendはoperation IDに対応するImplementationSpecを明示選択する。実装不足はcompile errorであり、
Python参照実装へ黙ってfallbackしない。次の`compile_kernel`はv0.4 adapter protocolである。

```python
class Backend(Protocol):
    name: str

    def compile_kernel(
        self,
        kernel: Kernel,
        context: CompileContext,
    ) -> CompiledKernel:
        ...
```

候補:

- PythonBackend
- CythonBackend
- CppBackend
- HybridBackend

## 2.2 backend指定

```python
plan = cw.compile(
    outputs,
    backend="python",
)
```

将来:

```python
backend = HybridBackend(
    preferred=["cpp", "cython", "python"],
)
```

Operation Backendの既定はfallbackなしとする。将来`PreferNative`等を明示指定した場合だけNode単位の
fallbackを許し、理由と選択結果をDiagnosticおよびPlanへ記録する。

BackendはNodeのアルゴリズム実装をcompileする。ExecutionPlan全体のready判定、rate cursor、frame history、buffer寿命はExecutorの責務であり、Backendへ含めない。Native KernelをPython Executorから呼ぶ構成と、NativeExecutorから呼ぶ構成を区別する。詳細は[10_Native_Executor設計方針.md](10_Native_Executor設計方針.md)を参照する。

v0.3ではBackendが返す`CompiledKernel`が任意で`NativeCompiledKernel`を実装できる。
ABI version、process model、workspace、flush、session ownership、native互換性はこのfactoryから
PortablePlanIRへ抽出する。これによりKernel宣言へ実装言語を埋め込まず、同じ宣言を
Python BackendまたはCython Backendでcompileできる。plain Python callableは選択Backendへ
渡さずPython Stageとして残るため、一つのPlan内でPython callbackとCython Kernelを混在できる。

v0.4で固定shape batchを処理するfactoryは`NativeBatchCompiledKernel`、sessionは
`NativeBatchKernelSession`を実装する。Executorは一つのread-only f64 memoryview、item count、
item shapeを一回だけ渡し、Kernelは`NativeValueBatch`を返す。出力shapeが入力shapeから決まる
Kernelは`NativeValueSchemaProvider`でcompile時に解決していた。Operation移行ではFFT、CBF等のshape
規則をBackend factoryからOperationSpecへ移し、Backend選択前にunifyする。IRへはresolver callableで
なくresolved Port schemaだけを記録する。

v0.4では`NativeRuntimeBindingProvider`が`NativeKernelRuntimeBinding`を生成する。これは新設計の
`ImplementationBinding`へ移行する先行実装である。PortablePlanIRは
ABI version、process model、binding slotだけを保持し、固定係数等のprocess-local定数はdtype、shape、
immutable bytesとして`ExecutionPlan.create_session()`時にCppExecutorへbindする。pointer、Python
class、allocatorはIRへ含めない。C++ runtimeは次のversion付きresolver tableからprocess modelを選ぶ。

| ABI ID | process model | parameter | output |
|---|---|---|---|
| `chronowire.kernel.identity_f64.v1` | `identity_f64` | なし | 入力shapeを保持 |
| `chronowire.reference.fixed_cbf_f64.v1` | `fixed_cbf_f64_frame` | read-only f64重み | beam × frame |
| `chronowire.reference.covariance_accumulator_f64_frame.v1` | `covariance_accumulator_f64_frame` | diagonal loading | channel × channel |
| `chronowire.reference.mvdr_weights_f64.v1` | `mvdr_weights_f64` | steering vector | channel |
| `chronowire.reference.apply_weights_f64_latest.v1` | `apply_weights_f64_latest` | なし | frame |

このidentity/固定CBF/MVDR resolver tableは、C++ Plan runtimeの意味論を先に検証するための
過渡的実装である。最終的には外部C ABI module tableの汎用dynamic bindingへ置換し、
新しいOperationの追加でCppExecutorやresolver tableを改修しない。既存ABIに載るDSP追加は
DSP package側のOperationSpec、Implementation、manifest entry、conformance testで完結させる。

固定CBFではCython Backendが生成したfactoryも同binding契約を提供するため、CppExecutorはPython Kernel
sessionを呼ばずにABI IDと係数だけからC++処理を選択する。未知ABI、process model不一致、parameter
shape不一致はNode、Port、binding slotを含む明示エラーとし、Pythonへ暗黙fallbackしない。

Backend本体の改修は、新dtype、可変shape、複数出力、0/複数Emission、Config packing、workspace、
flush/checkpoint、新ABI/process model、Implementation選択policyなど、Implementationを載せる器の
capabilityを拡張するときに限る。個別DSPアルゴリズムの追加はBackend改修の理由にしない。

## 2.3 backend境界

backendが異なるNode間では以下の変換が発生し得る。

- Python objectからnative buffer
- dtype変換
- device転送
- ownership移譲

compileは境界数と変換コストを考慮する。

## 2.4 C++移行性

Python固有にしない要素:

- Node依存関係
- Port/Edge ID
- Schedulerの基本状態機械
- logical time
- buffer index
- ExecutionPlan serialization
- Kernel ABI概念

# 3. Config

## 3.1 不変なスコープ付きConfig

```python
base_config = cw.Config(
    system={"fs": 32768},
)

beam_config = base_config.scope(
    beamformer={"bearing_deg": 20.0},
)

source = cw.Flow(source_impl, base_config)
beam = source.with_config(beam_config).map(beamform)
```

`scope()`は親Configを変更せず、新しいConfigを返す。Flow chainは現在のConfig scopeを参照し、分岐ごとに異なるscopeを持てる。

## 3.2 属性パスによる明示的なマージ

Configは階層を保持し、属性パス単位で親から子へ上書きする。

```python
base_config.system.fs
beam_config.system.fs
beam_config.beamformer.bearing_deg
```

辞書全体を暗黙に置換せず、指定されたleafだけを上書きする。

```python
child = base_config.scope(
    system={"block_size": 4096},
)

child.system.fs          # 親から継承
child.system.block_size  # 子scopeで追加
```

衝突する型、未知path、schema違反はscope生成時またはcompile時に診断する。

## 3.3 API

```python
config.require("system.fs")
config.get("system.fs", default=None)
config.has("beamformer.bearing_deg")
config.scope(system={"block_size": 4096})
config.to_dict(resolved=True)
```

属性代入、`set()`、run中の更新は提供しない。

## 3.4 GraphとPlanへの記録

各FlowはConfig scope IDを参照する。compile時にはNodeごとに以下をPlanへ記録する。

- scope IDと親scope ID
- OperationSpecが宣言したConfig scope内のleaf path。legacy Kernelでは既存config path
- pathごとの解決元scope
- compileに使用した解決値または再現可能なdigest

exportではConfig scopeの差分とNodeへの適用関係を表示し、設定がどの処理へ影響したか追跡可能にする。秘密値は値そのものをexportせず、redactionまたはdigestを使用する。

宣言Operationは`ConfigSpec(scope=..., fields=...)`でsubtreeと相対leaf依存を宣言し、そのConfigViewだけを
受け取る。plain Python callableだけは互換のためmap引数の`config_paths`を残す。宣言がないcallableは
scope全体への依存として扱い、部分的なcompile cache再利用を行わない。debug modeではruntimeの属性
アクセスを追跡し、宣言との差をDiagnosticにできる。

## 3.5 データ移動との境界

Configは実行中データを運ばない。時刻とともに変化する値はFlow、最新値参照はStateFlow、Operation内部の継続状態はOperationSession、診断はDiagnosticとして扱う。

```python
output = signal.map(
    process,
    steering=steering_flow,
    calibration=calibration_state,
)
```

この受け渡しはinput PortとEdgeとしてGraphへ記録される。Config scopeの変更は設定依存として記録するが、データEdgeとは区別する。

## 3.6 user intentを保持する

Configは実装詳細よりユーザー意図を優先する。

```python
config = config.scope(
    fft={"frequency_resolution_hz": 4.0},
)
```

FFT OperationSpecが`fs`と`df`から`nfft`をcompile時に解決する。Chronowire本体とBackend実装はDSP固有変換を推測しない。

## 3.7 schema

Chronowire本体は汎用Config機構だけを提供し、DSP Operation packageが任意のschemaを持つ。

- dataclass
- Pydantic
- 独自validator
- Protocol

## 3.8 runtime stateとthread safety

Configは不変なのでrun中の競合書込みを持たない。OperationSessionはPlanSessionが所有し、Scheduler管理下でのみ更新する。並列Stageで共有状態が必要な場合は、StateFlowによる順序と更新境界を明示する。
