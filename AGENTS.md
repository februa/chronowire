# AGENTS.md

このリポジトリは、論理時間に基づくストリーミング処理をFlow APIで記述し、Logical GraphからExecutionPlanへcompileするChronowireを扱う。

実装では、動作だけでなく、責務、型、時間区間、状態、buffer境界、劣化理由を後から説明できることを重視する。

## Chronowireの設計原則

1. Chronowireはspflowの後継や互換層ではなく、独立した明示compile型フレームワークである。
2. Flowは値ではなくGraph上のoutput Portを指す。Flow構築時にKernelを実行しない。
3. Configは不変なscopeとし、時変値や状態をConfig経由で運ばない。データ移動はFlowまたはStateFlowのEdgeとしてGraphへ記録する。
4. Graph、compiler、runtime、collector、Extension、Kernel protocolの責務を混ぜない。
5. `core`、`common`、`utils`、`misc`のような責務不明のpackage名を新設しない。
6. 公開APIは小さく保ち、DSPアルゴリズム、可視化、評価方式をChronowire本体へ含めない。
7. 0件、1件、複数件のEmissionを明示し、通常のlistやtupleを暗黙に展開しない。
8. 論理時間、interval、sequence、status、DiagnosticをEdgeやcollector通過時に失わない。
9. 安全なfallback、不十分な積分、観測可能な失敗は`DEGRADED`または`INVALID`として残す。安全に継続できない契約違反だけを例外にする。
10. 未完成の作業状態を公開しない。stateful Kernelはrun開始時にresetし、例外後に再実行可能な状態へ戻す。
11. 最適化前後、Backend間で値、interval、sequence、status、Diagnosticの意味を保つ。
12. 将来を予想した抽象化より、v0.1設計書で確定した契約の薄い実装を優先する。

## Python実装規約

- Python 3.11以上を対象とする。
- 公開class、公開function、公開methodには日本語docstringを書く。
- docstringには責務、引数、戻り値、例外、境界条件を記載する。
- コメントは処理の逐語説明ではなく、設計理由、不変条件、時間・状態の境界を説明する。
- 戻り値の型をflagで変えない。補助情報は固定shapeのdataclassへまとめる。
- `Any`はGraph境界や利用者値など必要な範囲に限定し、検証後は具体型へ絞る。
- `cast`で型エラーを隠さない。必要な場合は先に実行時検証を行う。
- mutableなdefault引数を使わない。
- 公開値は可能な限りfrozen dataclassまたはimmutable collectionで表す。
- 例外messageとDiagnosticにはNode、Port、interval、違反した契約を特定できる情報を含める。

## Cython実装規約

- `.pyx`を正本とし、自動生成された`.c`または`.cpp`を直接編集・commitしない。
- Python/C++境界でdtype、shape、stride、byte長、時刻分母、status値を検証してからtyped memoryviewまたはnative pointerへ変換する。
- hot loopではPython object、動的属性アクセス、Python callbackを作らず、C型、typed memoryview、contiguous bufferを使用する。
- `nogil`範囲を明示し、その中でPython C API、参照count操作、Python例外生成を行わない。Python処理が必要な箇所ではGILを明示的に取得する。
- `malloc`などの手動確保を使う場合は全return・例外経路で解放する。可能ならC++ RAIIまたは所有者が明確なbufferを使用する。
- C++例外を跨ぐ宣言には`except +`を指定し、`nogil`実行中の例外をPython境界で契約名付き例外へ変換する。
- `.pyx`の公開・Python可視APIを変更した場合は対応する`.pyi`を同時に更新し、Pyrightで利用側の型を検証する。
- 性能最適化で値、interval、sequence、status、Diagnostic provenanceを省略しない。

## C++実装規約

- C++17を基準とし、所有権はvalue、`std::vector`、RAII、smart pointerで表す。所有するraw pointerと手動`new`/`delete`を通常コードへ持ち込まない。
- process-globalなmutable状態を持たず、cursor、buffer、Kernel状態、collector状態はrun-local sessionが所有する。
- ABI境界ではdtype、shape、byte長、alignment、version、process modelを検証する。PortablePlanIRにpointer、Python object、実行時addressを保存しない。
- size積、時刻scale、tick加算、capacity計算では符号、narrowing、整数overflowを検査する。未検証の`static_cast`で警告を消さない。
- `std::span`相当の非所有viewと所有bufferを区別し、fan-out共有bufferはread-only寿命をconsumer完了まで保証する。
- C++例外をC ABIや`nogil`境界の外へ漏らさない。Cython adapterで捕捉できる型へ変換し、Node、Port、違反契約をPython側messageへ付加する。
- native runtime内でPython C APIを呼ばない。Python callbackはcompile済みStage境界としてのみ扱い、C++ Stageへ暗黙に混入させない。
- warningを放置せず、警告抑制attributeやcompiler optionを追加する前に設計上の原因を解消する。
- 最適化前後とPython/Cython/C++ Executor間で値、interval、sequence、status、Diagnosticを同値に保つ。

## テスト規約

- 正常例だけでなく、空入力、fan-out、0/1/複数Emission、EOF、interval不一致、collector overflow、Extension失敗、DEGRADED/INVALID伝播を試験する。
- 共通祖先が分岐ごとに二重実行されないことを試験する。
- bare Flowが値を保持しないこと、bounded collectorが上限を超えないことを試験する。
- compile warningとruntime errorを混同しない。
- 公開bugには最小再現testを追加する。

## 開発環境と検証

環境作成と実行にはuvを使用する。

```bash
uv sync --extra dev
uv run pytest
uv run pyright
uv run ruff check .
uv run ruff format --check .
uv run cython-lint --max-line-length 100 src/chronowire/*.pyx src/chronowire_reference/*.pyx
c++ -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror \
  -fsyntax-only src/chronowire/cpp_runtime.cpp
uv build
```

実装完了前にpytest、Pyright、Ruffをすべて通す。CythonまたはC++を変更した場合は、
`cython-lint`、C++のwarnings-as-errors構文検査、sdistからのnative wheel buildも通す。
hand-written C++ sourceを追加した場合は構文検査commandへ対象を追加する。依存を追加する場合は
`uv add`または`uv add --dev`を使い、`pyproject.toml`と`uv.lock`を更新する。

## 文書とskill

- Markdownを設計内容の正本とする。
- Word成果物が必要な場合は`.agents/skills/markdown-to-reviewed-word`を使用し、変換、監査、全page render確認まで行う。
- `beamforming-evaluation`はDSP Kernel package側のskillであり、Chronowire本体へは継承しない。

## 作業完了条件

- v0.1設計書と公開APIが一致している。
- 公開APIに日本語docstringがある。
- 安全な劣化結果が例外で失われない。
- pytest、Pyright、Ruffが成功する。
- Native変更ではcython-lint、C++ warnings-as-errors、`uv build`が成功する。
- 変更した文書のリンクと索引が整合する。
