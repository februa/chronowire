# Extension・export・診断設計

# 1. Extension

## 1.1 目的

Extensionは計算Nodeと分離された横断機能である。

- profiler
- logging
- trace
- snapshot
- config trace
- runtime metrics
- debug capture

保存やログを安易に`map(write_file)`としてGraphへ入れると、計算と副作用が混ざるため、観測用途ではExtensionを優先する。

## 1.2 compile-time観測契約とruntime binding

```python
observation = cw.observe(
    spectrum,
    extension_id="spectrum_snapshot",
    trigger=cw.EveryLogicalTime(period=5),
    priority=0,
)

plan = cw.compile(
    outputs,
    extensions=[observation],
)
session = plan.create_session(
    extension_bindings={
        "spectrum_snapshot": cw.Snapshot(path="snapshots/spectrum.jsonl"),
    }
)
result = session.run()
```

`observe()`が返す`ObservationSpec`はcompile-time契約であり、`extension_id`、観測Port、trigger、priority、failure/overflow policy、要求ABIを保持する。観測Portはrequired rootかつFusion境界となる。Extension観測だけを理由に`RunResult.outputs`へは追加しない。

`Extension`はprocess-local binding factoryであり、Flow、Port、triggerを知らない。`create_session()`がrun-localな`ExtensionSession`を生成する。`Snapshot`はJSONL recorder bindingであり、pathやPython objectをPortablePlanIRへ入れない。同じPlanまたはExecutionSessionを再実行しても、trigger、file、handlerの可変状態を持ち越さない。

`extension_id`は利用者が必ず明示し、Plan内で一意とする。`extension_id="spectrum_snapshot"`に対してCompilerは`binding_slot="extension:spectrum_snapshot"`を生成できるが、安定した観測契約IDとprocess-local注入slotは別field、別責務として扱う。

## 1.3 Trigger

v0.1は`Always`、event件数に基づく`Every`、論理時間に基づく`EveryLogicalTime(period, phase=0)`を提供する。`EveryLogicalTime`の境界は半開区間`[interval.start, interval.end)`へ所属する。境界が前Emissionの`end`と次Emissionの`start`に一致する場合は次Emissionだけが発火する。overlapするEmissionが同じ境界を覆っても、一つの観測契約ではその境界を最初に処理したEmissionだけが発火する。一つのEmission内に複数境界がある場合もcallbackは一回とし、次の未処理境界まで進める。

Extensionはcollectorと独立する。output collectorが`NoCollect`でもExtensionは対象PortのEmissionを観測できる。`Snapshot(include_degraded=True)`は`DEGRADED`および`INVALID`なEmission、対応Diagnostic、論理時間区間を保存する。

## 1.4 Binding検証

`ExecutionPlan.create_session(extension_bindings=...)`は実行前に次を検証する。

- 必須`extension_id`のbinding不足
- Planに存在しない、または未使用のbinding
- `Extension.create_session()`を持たないbinding種別
- `ObservationSpec`とbindingのABI version不一致

違反は明示例外とし、messageへNode、Port、`extension_id`、binding slot、違反契約を含める。`extension_id`重複はbinding時まで遅延せずcompile errorとする。

## 1.5 実行とbackpressure

v0.1のExtension callbackはScheduler threadで同期実行し、決定性を優先する。公開failure/overflow policyは`FAIL`だけを実装する。handler例外は`ExtensionExecutionError`としてrunを停止し、ID、slot、Node、Port、callback、policyをmessageへ残す。

非同期handlerを追加する場合は、重いI/OをExtension内部のbounded queueへ渡し、次のpolicyを明示する。

```text
BLOCK
    Schedulerを待機させ、取りこぼさない

DROP_OLDEST / DROP_NEWEST
    実行を継続し、drop件数をDiagnosticへ記録する

FAIL
    ExtensionExecutionErrorとしてrunを停止する
```

無制限queueは禁止する。Extensionが失敗した場合に計算を継続するか停止するかもExtensionごとに宣言する。

`PortBuffer.max_items`はCompiler/Schedulerが決める計算経路のcapacityであり、Extensionの保存件数ではない。観測履歴はExtension内部のbounded window、`RunResult`の保持件数はCollectorで個別に宣言する。

## 1.6 最適化への影響

観測PortはFusion境界となる。

初期版ではExtension観測地点を跨ぐFusionを禁止する。

# 2. graph_info()

```python
info = flow.graph_info()
```

論理Graphの読み取り専用情報を返す。

用途:

- テスト
- 独自解析ツール
- IDE連携
- export基盤
- compile前診断

# 3. export()

## 3.1 Flow.export()

論理Graphを出力する。

```python
flow.export("graph.json")
flow.export("graph.dot")
flow.export("graph.mmd")
flow.export("graph.svg")
```

## 3.2 Plan.export()

compile後のExecutionPlanを出力する。

```python
plan.export("plan.json")
plan.export("plan.dot")
plan.export("plan.svg")
plan.export("plan.md")
```

Plan側には以下を含める。

- required Node
- compiled Node
- fused group
- batched group
- parallel stage
- buffer allocation
- backend assignment
- output index
- observation boundary
- diagnostics
- Config scopeとNodeごとの参照path
- collector policyとdrop count
- Emission status集計

## 3.3 表示名

`.named()`を使わないため、次を組み合わせて表示する。

```text
Node 17
Kernel: BeamformerKernel
Output[0]
Port 18
```

Python callable:

```text
Node 5
Callable: preprocess
Module: user_pipeline
```

同名関数でもNode IDで区別できる。

## 3.4 JSON例

```json
{
  "schema_version": "1.0",
  "kind": "execution_plan",
  "outputs": [
    {
      "index": 0,
      "port_id": 18,
      "node_id": 17
    }
  ],
  "nodes": [],
  "edges": [],
  "stages": [],
  "buffers": [],
  "diagnostics": []
}
```

## 3.5 SVG/PNG

```text
GraphInfoまたはPlan IR
    ↓
DOT
    ↓
Graphviz
    ↓
SVG/PNG
```

Graphviz未導入時はDOTまたはMermaid exportを案内する。

# 4. 例外

## 4.1 例外階層

```text
ChronowireError
├ GraphError
│ ├ GraphMismatchError
│ ├ InvalidEdgeError
│ ├ CycleError
│ └ InvalidPortError
├ CompileError
│ ├ EmptyOutputError
│ ├ DuplicateOutputError
│ ├ UnsupportedKernelError
│ ├ TimeSemanticsError
│ └ BackendCompileError
├ ChronowireRuntimeError
│ ├ SourceError
│ ├ KernelExecutionError
│ ├ SynchronizationError
│ ├ BufferOverflowError
│ └ DeadlockError
├ ConfigError
│ ├ MissingConfigError
│ ├ InvalidConfigError
│ └ ConfigMergeError
├ ExtensionError
└ ExportError
```

## 4.2 compile検証

- outputsが空でない
- 全要素がFlowまたはOutputSpec
- 全Flowが同一Graph
- output Portが存在
- DAGである
- required Node入力が接続済み
- Kernelがbackend対応
- 時間意味論が整合
- Sourceへ到達可能
- Config必須値が存在

未定義timebase、負のduration、Kernelの矛盾した時間変換宣言など、時間意味論自体の契約違反は`TimeSemanticsError`とする。複数入力のintervalが一致しない可能性はcompile errorにせず、`POSSIBLE_INTERVAL_MISMATCH` warningとする。

## 4.3 warning

- fallback to Python backend
- 観測境界によるFusion阻害
- bounded dynamic bufferの明示上限使用
- 非整数rate比
- merge入力のinterval不一致可能性 (`POSSIBLE_INTERVAL_MISMATCH`)
- frame末尾drop
- compile対象外の大きな経路
- Config scopeの未使用override
- 劣化結果の発生
- collectorまたはExtension queueでのdrop

静的上限を証明できず、明示`max_items`もbackpressureもないruntime bufferはwarningではなくcompile errorとする。`REALTIME_INGRESS`のdropは実行継続可能な外部入力欠落であり、`INPUT_OVERRUN` Diagnosticとして記録する。

## 4.4 Diagnostic

```python
@dataclass(frozen=True)
class Diagnostic:
    severity: Severity
    code: str
    message: str
    node_id: int | None = None
    port_id: int | None = None
    hint: str | None = None
    interval: LogicalInterval | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
```

Diagnosticは例外専用ではない。`DEGRADED`または`INVALID`なEmissionには、fallback種別、観測数、必要条件、利用上の制約などの機械可読なdetailsを関連付ける。

## 4.5 劣化結果と例外の境界

安全な値または安全な無効結果を生成できる場合は、例外でrunを打ち切らない。

```text
DEGRADED Emission:
  node_id: 17
  interval: [1.500000, 1.531250)
  code: INSUFFICIENT_INTEGRATION
  details:
    observed_snapshots: 4
    recommended_snapshots: 16
    fallback: fixed_cbf
```

ExtensionとcollectorはこのEmissionを通常の時間順序で受け取る。後段Kernelはstatus policyに従って受理、伝播、別fallback生成のいずれかを行う。

## 4.6 Kernel実行例外

```text
KernelExecutionError:
  node_id: 17
  kernel: BeamformerKernel
  interval: [1.500000, 1.531250)
  backend: python
  cause: ValueError("steering vector shape mismatch")
```

元例外は`__cause__`として保持する。

例外は、shape契約違反、破損したBackend状態、安全なfallbackを作れない実装障害など、継続が安全でない場合に限定する。

## 4.7 stalled NodeとDeadlock検出

exact mergeが必要intervalを待つ場合、Schedulerは不足経路だけを進める。producer frontierまたは先頭itemが必要intervalを通過し、今後生成不能と確定した時点で`STALLED_EXACT_MERGE`を記録する。これはSourceを先行実行してbufferを満たした後に検出するものではない。

以下が成立した場合にDeadlockまたは同期不能と判定する。

- 未処理Nodeあり
- ready Nodeなし
- SourceがEOFまたは要求不能
- buffer状態が変化しない

診断には待機Node、入力Port、不足intervalを含める。

`STALLED_EXACT_MERGE`は該当Nodeへの需要を停止するが、独立した観測終端は継続する。実行可能な需要が残らない場合はrunを終了し、`RunResult.completed`とDiagnosticから未完了経路を判別できるようにする。

## 4.8 realtime input Diagnostic

`REALTIME_PUSH`のbuffer overflowでは、少なくとも次を記録する。

- `INPUT_OVERRUN`
- Source ID、Port ID
- 欠落interval
- 今回および累積のdrop件数
- `max_items`とoverflow policy

drop後に最初に配送するEmissionは`DEGRADED`とし、同じDiagnosticを関連付ける。次のEmissionがない場合でもDiagnosticはExtensionと`RunResult`へ残す。
