# Chronowire

Chronowireは、論理時間付きストリームをFlow APIでLogical Graphとして構築し、明示的にcompileして実行するPythonフレームワークである。

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

v0.1とv0.2のrelease gateは[doc/chronowire_design/12_v0.1_v0.2リリース方針.md](doc/chronowire_design/12_v0.1_v0.2リリース方針.md)に定める。Native Executorはv0.3以降とする。
