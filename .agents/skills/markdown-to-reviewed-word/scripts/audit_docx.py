#!/usr/bin/env python3
"""生成DOCXの固定表geometry、画像alt text、placeholderを監査する。"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS = {"w": WORD_NAMESPACE, "wp": DRAWING_NAMESPACE}


def _attribute(namespace: str, name: str) -> str:
    """ElementTree用のnamespace付きattribute名を返す。"""
    return f"{{{namespace}}}{name}"


def audit_docx(path: Path) -> tuple[str, ...]:
    """DOCX構造を監査する。

    Args:
        path: 監査対象DOCX。

    Returns:
        問題説明のtuple。空tupleなら合格。

    Raises:
        FileNotFoundError: DOCXが存在しない場合。
        zipfile.BadZipFile: DOCX packageが壊れている場合。
    """
    if not path.exists():
        raise FileNotFoundError(path)
    issues: list[str] = []
    with zipfile.ZipFile(path) as package:
        document_xml = package.read("word/document.xml")
    root = ET.fromstring(document_xml)

    text = "".join(element.text or "" for element in root.findall(".//w:t", NS))
    for marker in ("TODO", "[画像未生成", "[PLACEHOLDER"):
        if marker in text:
            issues.append(f"placeholder text remains: {marker}")

    headings = root.findall(".//w:pStyle[@w:val='Heading1']", NS)
    if not headings:
        issues.append("Heading 1 is missing")

    for table_index, table in enumerate(root.findall(".//w:tbl", NS), start=1):
        width = table.find("./w:tblPr/w:tblW", NS)
        grid_columns = table.findall("./w:tblGrid/w:gridCol", NS)
        if width is None or width.get(_attribute(WORD_NAMESPACE, "type")) != "dxa":
            issues.append(f"table {table_index}: tblW is not fixed DXA")
            continue
        expected = int(width.get(_attribute(WORD_NAMESPACE, "w"), "0"))
        grid_widths = [
            int(column.get(_attribute(WORD_NAMESPACE, "w"), "0"))
            for column in grid_columns
        ]
        if sum(grid_widths) != expected:
            issues.append(f"table {table_index}: tblGrid sum does not match tblW")
        for row_index, row in enumerate(table.findall("./w:tr", NS), start=1):
            cell_widths = [
                int(cell.get(_attribute(WORD_NAMESPACE, "w"), "0"))
                for cell in row.findall("./w:tc/w:tcPr/w:tcW", NS)
            ]
            if cell_widths != grid_widths:
                issues.append(f"table {table_index} row {row_index}: tcW does not match tblGrid")

    for image_index, properties in enumerate(root.findall(".//wp:docPr", NS), start=1):
        if not properties.get("descr", "").strip():
            issues.append(f"image {image_index}: alt text is missing")

    return tuple(issues)


def main() -> None:
    """CLIからDOCXを監査し、問題があれば終了code 1を返す。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", type=Path)
    arguments = parser.parse_args()
    issues = audit_docx(arguments.docx)
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        raise SystemExit(1)
    print(f"OK: {arguments.docx}")


if __name__ == "__main__":
    main()
