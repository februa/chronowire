---
name: markdown-to-reviewed-word
description: Convert Japanese or English technical Markdown into a polished, editable DOCX and verify its structure and every rendered page. Use when Codex must publish design documents, evaluation reports, equations, figures, code blocks, or wide tables as a review-ready Word document while keeping Markdown as the source of truth.
---

# Markdown to Reviewed Word

Markdownを正本として保ち、編集可能なWord文書と全ページの目視確認結果を生成する。
単なる拡張子変換ではなく、数式、図、表、見出し、改ページの意味をWord構造へ移す。

## 実行手順

1. 入力Markdownと参照画像を読む。変換不能なHTML、入れ子list、未画像化Mermaidがないか確認する。
2. Wordで横長になる表の直前へ`<!-- word-table: records -->`を置き、行単位の縦型recordへ変換する。
3. Mermaidは内容を確認してPNGまたはSVGへ描画し、通常のMarkdown画像として参照する。Mermaid fenceを未変換のまま残さない。
4. bundled workspace Pythonで`scripts/markdown_to_word.py`を実行する。
5. `scripts/audit_docx.py`で表geometry、画像alt text、見出し、placeholderを監査する。
6. `scripts/render_docx_pages.py`で全pageをPNGへ変換する。
7. 全PNGを100%相当で確認する。文字切れ、重なり、表の窮屈さ、欠落glyph、図とcaptionの分離、不自然な空白を直して再生成する。
8. 最新の監査と全page確認が通ったDOCXだけを成果物として渡す。

## 基本コマンド

`python`にはシステムPythonではなく、Codex workspace dependenciesが返すPythonを使う。

```bash
python scripts/markdown_to_word.py input.md output.docx
python scripts/audit_docx.py output.docx
env TMPDIR=/private/tmp python scripts/render_docx_pages.py output.docx --output-dir /tmp/word-pages
```

生成例を変換テストに使う場合は`--title`、`--subtitle`、`--font`を必要なときだけ指定する。
既定fontはmacOSの`Arial Unicode MS`である。レンダリング時は対応font fileを一時LibreOffice profileへ自動供給する。別OSではWordとLibreOfficeの双方で実在する日本語fontを`--font`へ指定し、必要ならレンダリングへ`--font-file`も渡す。
数式を構造化OMMLへ変換することが必須なら`--require-structured-equations`を指定する。
既定styleは技術設計書向けの`compact_reference_guide`相当で、Letter、1 inch margin、本文11 pt、固定9360 DXA表を使う。

## Markdown契約

- 最初の`#`見出しを文書titleとして使う。`--title`指定時は指定値を優先する。
- `##`、`###`、`####`をWordのHeading 1、2、3へ移す。
- `\[ ... \]`を編集可能なWord数式へ変換する。`latex2mathml`がない場合もLaTeX文字列を数式領域へ残し、欠落させない。
- Markdown画像はsourceからの相対pathで解決し、alt textをWordへ設定する。
- pipe tableは固定幅のWord表へ変換する。比較ではなく長文recordなら`word-table: records`を使う。
- fenced codeは背景付きcode blockにする。Mermaid fenceは自動的にcode扱いせず、画像化を要求する。
- `<!-- pagebreak -->`を明示改pageとして扱う。
- 未対応構文は意味を推測して捨てず、入力位置付きエラーにする。

詳細な変換判断は[references/conversion-contract.md](references/conversion-contract.md)を読む。

## 固有文書との境界

汎用scriptへ、章番号や特定方式名に依存する処理を追加しない。
固有の表紙、方式図、評価summaryを追加する場合は、汎用scriptをimportする薄いrepository toolを作る。
repository固有の表紙、方式図、summaryが必要な場合は、`tools/`に汎用scriptをimportする薄いbuilderを置き、Markdown変換本体と固有要素を分離する。

## 完了条件

- Markdownの章、図、表、数式が欠落していない。
- `audit_docx.py`が成功する。
- 生成された全page PNGを確認済みである。
- 最新renderに文字切れ、重なり、欠落glyph、壊れた表、孤立captionがない。
- QA用PNGや一時PDFを成果物として渡さない。
