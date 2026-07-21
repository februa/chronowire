# Chronowire 公開API設計

## 1. import

```python
import chronowire as cw
```

トップレベルの主要公開候補は次とする。

```python
cw.Flow
cw.Config
cw.compile
cw.ExecutionPlan
cw.OutputSpec
cw.OutputResult
cw.RunResult
cw.Emission
cw.EmissionStatus
cw.skip
cw.emit_many
cw.Source
cw.RealtimeSource
cw.Kernel
cw.CompiledKernel
cw.CompiledKernelSession
cw.GraphInfo
```

内部Graph、Scheduler、Node、Edge、Portは原則としてトップレベル公開しない。

## 2. Flow生成

基本形:

```python
flow = cw.Flow(source, config)
```

概念的シグネチャ:

```python
class Flow(Generic[T]):
    def __init__(
        self,
        source: Source[T] | Iterable[T],
        config: Config | Mapping[str, Any] | None = None,
    ) -> None:
        ...
```

Sourceと有限Iterableの判別が曖昧になる場合は将来Factoryを追加する。

```python
Flow.from_source(...)
Flow.from_iterable(...)
```

ただし主APIはconstructorとする。

## 3. map()

```python
out = flow.map(process)
```

追加引数も利用できる。

```python
out = signal.map(
    combine,
    reference=reference_flow,
    gain=0.5,
)
```

引数の解釈:

| 種別 | Graph上の意味 |
|---|---|
| 通常の値 | 定数Node parameter |
| Flow | 同期入力Edge |
| StateFlow | latest state入力 |
| Config | Nodeが参照する不変な設定scope |

### 3.1 Config自動注入

以下の両方を許す。

```python
def process(x):
    ...

def process_with_config(x, config):
    ...
```

ユーザーはどちらも同じように登録する。

```python
flow.map(process)
flow.map(process_with_config)
```

`map(process, config=config)`は提供しない。scopeを変更する場合は`flow.with_config(config)`を明示し、Nodeがどのscopeを参照するかをGraphへ記録する。

シグネチャ解析はcompile時に一度だけ行い、runごとに`inspect.signature()`を呼ばない。

### 3.2 データ移動と設定を分ける

Configは不変な設定の解決にだけ使用する。時変パラメータ、状態更新、経路間で受け渡す値はFlowまたはStateFlow引数にする。

```python
steered = signal.map(
    steer,
    bearing=bearing_flow,
    calibration=calibration_state,
)
```

この例では`bearing`は同期Edge、`calibration`はlatest state EdgeとしてGraphに記録される。Configを介した暗黙のデータ移動は禁止する。

### 3.3 0/1/複数Emission

v0.1の公開Flow APIは単一output Portとし、Python callableの戻り値を次のように解釈する。

```python
return value                 # 1 Emission
return cw.skip()             # 0 Emission
return cw.emit_many(values)  # 複数Emission
```

通常のlistやtupleは一つの値として扱い、暗黙に複数Emissionへ展開しない。`emit_many()`の各値には、Kernelが明示的な時間変換を宣言しない限り入力intervalを引き継ぐ。

`emit_many()`を返せるKernelまたはPython callableは、一回の呼出しで生成するEmission上限をdescriptorの`max_items`へ宣言する。単一値または`skip()`だけを返す処理のdefaultは`max_items=1`とする。上限を超えた呼出しは契約違反であり、その呼出しの出力を一件も公開しない。plain callableへ上限を付与する具体的なdecoratorまたはadapter APIは、Kernel descriptor実装時に一つへ統一する。

### 3.4 Config参照path

Python callableがConfigを受け取る場合、主APIでは`map()`に参照pathを明示する。

```python
out = flow.map(
    process_with_config,
    config_paths=("system.fs", "beamformer.bearing_deg"),
)
```

再利用可能なKernel用decoratorは補助APIとし、v0.1の必須機能にはしない。`config_paths`を省略したcallableはConfig scope全体へ依存するとみなす。

## 4. frame()

```python
frames = flow.frame(
    size=1024,
    hop=512,
)
```

候補シグネチャ:

```python
def frame(
    self,
    size: int,
    *,
    hop: int | None = None,
    axis: int = -1,
    pad_end: bool = False,
) -> Flow[Frame[T]]:
    ...
```

- `hop=None`なら`hop=size`
- frameはバッファリング機能でありSTFTではない
- window処理やFFTはKernel側の責務
- 出力は入力の論理区間を保持する

## 5. rate()

```python
out = flow.rate(100.0)
```

`rate()`は論理的な出力周期・起動周期を制御する。

数値補間、デシメーションフィルタ、アンチエイリアス処理はChronowire coreの責務ではない。必要な数値処理はKernelへ委譲する。

候補policy:

```text
AUTO
HOLD
AGGREGATE
REQUEST
```

初期版では機能範囲を限定する。

v0.1は`HOLD`だけを実装する。入力Emissionのinterval内にある発火時刻ごとに同じ値を出力し、出力intervalを`1 / frequency_hz`の有理周期に置き換える。これは値の補間ではなく、SchedulerがNodeを起動する論理時刻の指定である。

rate/frameの正規形は`flow.rate(...).frame(...)`とする。`frame(...).rate(...)`は、完成済みframeをHOLDして複製するか、一部のframeを使わない経路を作れるためcompile errorとする。preserve MAPを挟んでも旧frame格子は継続するため同じく拒否する。

frame列を入力に取る数値resamplingが必要な場合は、外部Kernelを`time_transform="explicit"`としてFlowへ置き、旧格子を終了する。その出力を直接frame化せず、`rate(...).frame(...)`で新しい有理周期とframe境界を宣言する。RATEを含む完全同期入力はdurationとperiodの一致を静的に証明できなければcompile errorとし、runtimeのdropやstallで帳尻を合わせない。

下流にRATE Nodeを持つgenerated Sourceについては、compileがSourceごとの最短rate周期を求め、その周期を`SourceRequest.duration`に使用する。

## 6. graph_info()

```python
info = flow.graph_info()
```

Flowが属するLogical Graph全体の不変スナップショットを返す。

```python
@dataclass(frozen=True)
class GraphInfo:
    nodes: tuple[NodeInfo, ...]
    edges: tuple[EdgeInfo, ...]
    ports: tuple[PortInfo, ...]
    sources: tuple[SourceInfo, ...]
    diagnostics: tuple[Diagnostic, ...]
```

Graph本体の可変オブジェクトは返さない。

## 7. export()

FlowとExecutionPlanの両方が`export()`を持つ。

```python
flow.export("logical_graph.json")
plan.export("execution_plan.json")
```

`draw()`、`report()`、`to_plan_info()`は設けない。

対応候補:

- JSON
- DOT
- Mermaid
- SVG
- PNG
- Markdown
- Plain text

拡張子からformatを推定する。

## 8. chronowire.compile()

compileはFlowメソッドではなく、パッケージトップレベルの公開APIとする。

```python
plan = cw.compile([
    cw.output(beam, collector=cw.Latest()),
    cw.output(covariance, collector=cw.Bounded(max_items=64)),
])
```

候補シグネチャ:

```python
def compile(
    outputs: Sequence[Flow[Any] | OutputSpec[Any]],
    *,
    backend: str | Backend = "python",
    optimization: OptimizationLevel = OptimizationLevel.DEFAULT,
    extensions: Sequence[ObservationSpec] = (),
) -> ExecutionPlan:
    ...
```

### 8.1 compile規則

- `outputs`は順序付きリストまたはSequence
- 空Sequenceは禁止
- 単一出力でも`cw.compile([flow])`
- 全Flowは同一Graphに属する
- 指定FlowはGraph途中でもよい
- 指定FlowがPlan上の観測終端になる
- bare Flowは`collector=NoCollect()`として扱い、全値を暗黙には保持しない
- OutputSpecは観測終端とcollector policyを明示する
- 未指定の固有経路はPlanへ含めない
- 共通祖先は一回だけ実行する
- 結果順はcompile指定順と一致する
- `.named()`は要求しない
- `extensions`にはhandler実体ではなく`cw.observe()`が返す`ObservationSpec`を渡す
- `extension_id`重複はcompile errorとする
- 観測Portはrequired rootおよびFusion境界とする

Extension実体はcompile後にprocess-local bindingとして注入する。

```python
observation = cw.observe(
    spectrum,
    extension_id="spectrum_snapshot",
    trigger=cw.EveryLogicalTime(period=5),
)
plan = cw.compile(outputs, extensions=[observation])
session = plan.create_session(
    extension_bindings={
        "spectrum_snapshot": cw.Snapshot(path="spectrum.jsonl"),
    }
)
result = session.run()
```

`extension_id`は安定した観測契約IDであり、`extension:spectrum_snapshot`のようなbinding slotとは同一fieldにしない。missing、unknown、unused、binding種別、ABI不一致は`create_session()`で明示例外にする。

## v0.2公開API

継続実行は一回実行の`ExecutionSession`と区別し、次の明示lifecycleを持つ。

```python
session = plan.create_plan_session(options=cw.RuntimeOptions(max_scheduler_steps=1000))
session.start()
session.run_until(10)
session.run_until(20)
result = session.close()
```

`run_until()`は境界を単調増加させ、Kernel、FRAME、RATE、buffer、collector、Extension trigger状態を保持する。budget終了時だけ同じ境界を再指定できる。`close()`はRealtime受付停止後にdrainし、`cancel()`は未処理値を破棄して`SESSION_CANCELLED`を残す。

追加Flow入力の同期は`flow.synchronize()`で明示する。referenceは常にMAPの主入力index 0、tie-breakは最小sequenceとする。`MissingInputPolicy.STALL`はNodeを診断停止し、`SKIP`は該当referenceだけを破棄する。

複数outputは通常tupleを暗黙展開せず、`map_outputs(..., output_count=N)`と`kernel_outputs(...)`を組み合わせる。外部制御値は`main.state_source(source)`で同じGraphへSOURCE Portを追加し、LATEST Edgeとして渡す。

plain callableの契約は`callable_kernel()`へまとめる。v0.2ではtime transformは`preserve`だけを提供し、`max_items`、`accepts_invalid`、gap後の`RESET`/`CONTINUE`をIRへ固定する。

### 8.2 途中Flowの指定

```python
base = source.map(preprocess)
beam = base.map(beamform)

plan = cw.compile([
    base,
    beam,
])
```

この場合、`base`と`beam`の両方が観測境界になる。ただしbare Flowの値は保持されない。値が必要な地点だけ`cw.output(base, collector=...)`を指定する。観測境界を跨ぐFusionは原則として禁止される。

### 8.3 OutputSpecとcollector

```python
cw.output(flow, collector=cw.NoCollect())
cw.output(flow, collector=cw.Latest())
cw.output(flow, collector=cw.Bounded(max_items=128))
cw.output(flow, collector=cw.Sink(write_item))
```

- `NoCollect`: 実行のみ。値を保持しない
- `Latest`: 最新Emissionだけ保持
- `Bounded`: 上限件数まで保持し、overflow policyを明示する
- `Sink`: Emissionを外部処理へ逐次渡す

全件保持collectorは初期公開APIに含めない。必要なら利用者が明示的なSinkとして実装する。

## 9. ExecutionPlan

```python
results = plan.run(duration=60.0)
plan.export("plan.json")
```

概念的API:

```python
class ExecutionPlan:
    def run(
        self,
        *,
        duration: float | None = None,
    ) -> RunResult:
        ...

    def export(
        self,
        path: str | Path,
        *,
        format: str | None = None,
    ) -> None:
        ...
```

ExecutionPlanはcompile後に不変とする。

## 10. run()戻り値

```python
run_result = plan.run(duration=60.0)
beam_result, covariance_result = run_result.outputs
```

`RunResult.outputs`はcompile指定順と対応する。値の件数はcollector policyで決まり、全件保持は保証しない。

候補型:

```python
@dataclass(frozen=True)
class OutputResult(Generic[T]):
    emissions: Sequence[Emission[T]]
    collector: CollectorInfo
    received_count: int
    dropped_count: int
    logical_start: LogicalTime | None
    logical_end: LogicalTime | None
    metadata: Mapping[str, Any]

@dataclass(frozen=True)
class RunResult:
    outputs: tuple[OutputResult[Any], ...]
    diagnostics: tuple[Diagnostic, ...]
    status_counts: Mapping[EmissionStatus, int]
    completed: bool
```

`received_count`は実行中に到着した総数、`dropped_count`はcollectorが保持しなかった数である。Extensionによる保存件数は各Extensionの結果・診断へ記録する。

## 11. Source protocol

Sourceは制御可能性によってpullとrealtime pushを別protocolにする。同じobjectに両方の意味を持たせず、PortablePlanIRのSource descriptorへ`PULL_CONTROLLED`または`REALTIME_PUSH`を記録する。

### 11.1 Pull-controlled Source

```python
class Source(Protocol[T]):
    def read(
        self,
        request: SourceRequest,
        config: Config,
    ) -> SourceBatch[T]:
        ...
```

```python
@dataclass(frozen=True)
class SourceRequest:
    logical_start: LogicalTime
    duration: LogicalDuration
```

有限Source:

- EOFを返せる
- `run()`でEOFまで実行できる

生成Source:

- `run(duration=...)`が必要
- Schedulerが区間を要求する

有限IterableもExecutorが`next()`の時点を制御できるため、内部adapterで`PULL_CONTROLLED`として扱う。

### 11.2 Realtime push Source

```python
class RealtimeSource(Protocol[T]):
    def start(
        self,
        receiver: RealtimeReceiver[T],
        config: Config,
    ) -> RealtimeSourceSession:
        ...

class RealtimeReceiver(Protocol[T]):
    def publish(self, emission: Emission[T]) -> None: ...
    def close(self) -> None: ...
    def fail(self, error: BaseException) -> None: ...
```

`RealtimeReceiver`はExecutorが提供し、thread-safeな`REALTIME_INGRESS`へ配送する。Source bindingは`max_items`とoverflow policyを必須とする。v0.1のdefaultは`DROP_OLDEST`であり、`DROP_NEWEST`は明示指定時だけ許可し、`BLOCK`は禁止する。

公開constructorは引き続き`Flow(source, config)`を主APIとする。pullとrealtimeの判別はruntime-checkable protocolまたは明示adapterで一度だけ行い、GraphへSource modeを記録する。曖昧性が生じる場合は黙って推測せず、明示adapterを要求する。

## 12. `.named()`を設けない理由

- Graphの正しさに名前は不要
- 内部Port IDで一意に識別できる
- Python変数名は取得不能または不安定
- 同名関数が複数Nodeに使われても問題ない
- Graph管理をユーザーへ露出しない思想を維持できる

export時にはNode ID、Port ID、Kernel表示名、Output indexを使用する。
