# 変換契約

## 正本と成果物

Markdownを内容の正本、DOCXをレビュー・配布用成果物とする。Word側だけを修正せず、内容変更はMarkdownまたは生成scriptへ戻す。

日本語fontはWord側の表示名だけで判断せず、DOCXを変換するLibreOffice環境にも同じfont fileを供給する。macOS既定は`Arial Unicode MS`とし、レンダリングscriptが一時LibreOffice profileへfont fileをcopyする。別OSでは`--font`で利用可能なCJK font名へ切り替え、`--font-file`で対応fileを渡す。

## 要素別の判断

| Markdown要素 | Word表現 | 境界条件 |
|---|---|---|
| 最初のH1 | plain paragraphのtitle | Word組込みTitle styleへ依存しない |
| H2--H4 | Heading 1--3 | H5以降は未対応としてエラー |
| paragraph | Normal style | 改行だけで別paragraphにしない |
| bullet / number | Word list style | 一段だけ。入れ子は事前に構造を直す |
| pipe table | 固定DXA table | 全列幅合計9360 DXA、header rowを反復 |
| record table | 見出しとlabel/value本文 | 直前に`<!-- word-table: records -->`を置く |
| image | inline imageとcaption | source相対path、alt text必須 |
| display equation | Word OMML | `\[`と`\]`を独立行にする |
| fenced code | 一列table内のmonospace text | codeの改行を保持する |
| Mermaid | 事前生成画像 | 未変換fenceはエラーにする |
| page break | Word page break | `<!-- pagebreak -->`を使う |

## 図

Mermaid sourceは図の正本としてMarkdown内に保持してよいが、Word生成前に内容を検証してPNGまたはSVGを生成する。画像参照をMermaid fenceの近くに置き、同じ図を二重掲載しない。Wordへ挿入した画像にはMarkdown alt textを設定する。

## 表

比較可能な反復fieldだけを表にする。説明文が大半を占める横長表はrecord形式へ変換する。列幅はheaderだけでなく全cellの文字量から配分し、短い識別列を狭く、説明列を広くする。固定行高は使わない。

## 数式

display数式は`latex2mathml`が利用可能ならMathMLを経由して構造化OMMLへ変換する。変換器がない場合や扱えない演算子でも式そのものを消さず、OMMLの文字runとしてLaTeXを残す。構造化数式が成果物要件なら`--require-structured-equations`を指定し、fallbackを早期エラーにする。数式の正しさはMarkdownとWordの双方で確認する。

## 視覚確認

DOCX生成成功を完成条件にしない。専用LibreOffice profileでPDF化し、各pageをPNGへ変換する。全pageについて次を確認する。

- 見出し直後の不自然な改pageがない。
- 表cellの文字が罫線へ密着またはclipしていない。
- 図とcaptionが離れていない。
- 日本語、数式、記号に欠落glyphがない。
- header/footerが本文へ重なっていない。
- 大きな空白が表や図の配置失敗によるものではない。

## repository固有拡張

表紙、冒頭summary、方式固有flow図などは汎用parserへ条件分岐として埋め込まない。repositoryの`tools/`に薄いbuilderを置き、汎用変換関数を利用して前後へ固有要素を追加する。
