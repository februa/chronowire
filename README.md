# Chronowire

Chronowireは、論理時間付きストリームをFlow APIでLogical Graphとして構築し、明示的にcompileして実行するPythonフレームワークである。

## 現在のv0.1 baseline

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

この一覧は現在動作するbaselineであり、v0.1 release完了を意味しない。v0.1ではさらに、gap後のexact merge再同期、conformance trace、残るPortablePlanIR fieldを実装する。

v0.1のmergeは完全interval一致とlatest stateに限定する。不一致の可能性はcompile warningとし、runtimeは必要intervalを生成できない経路を`STALLED_EXACT_MERGE`として停止する。

`rate()`は論理時間上の発火周期を有理数で管理する。v0.1のHOLD policyは数値補間を行わず、発火時点を含む入力intervalの値を出力する。generated Sourceへのrequest幅も、下流で必要な最短rate周期から決定する。

同一EmissionはExtension、観測終端collector、下流consumerの順で配送する。collector overflow時も、Extensionは対象Emissionを先に保存できる。

## 実行例

chunk入力をsampleへ展開し、rate、frame、EOF padding、固定CBFへ流す例を実行できる。

```bash
uv run python -m examples.streaming_cbf
```

## Install

現在の開発版はGit tagを指定してrevisionを固定できる。

```bash
uv add "chronowire @ git+https://github.com/februa/chronowire.git@v0.1.0.dev0"
```

```bash
python -m pip install "chronowire @ git+https://github.com/februa/chronowire.git@v0.1.0.dev0"
```

`0.1.0.dev0`はv0.1 release gate前の開発版であり、0.xの間はGitHub tagからinstallする。PyPIへは、Python/C++ Executor、継続streaming、評価例、API usabilityを確認したv1.0から正式公開する。release方針は[RELEASING.md](RELEASING.md)を参照する。

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

v0.1のrelease gateとv0.2の確定範囲は[doc/chronowire_design/12_v0.1_v0.2リリース方針.md](doc/chronowire_design/12_v0.1_v0.2リリース方針.md)に定める。v0.2は継続`PlanSession`、拡張同期、複数output Port、明示PortablePlanIR binding、長時間streaming観測を扱い、Native Executorはv0.3以降とする。
