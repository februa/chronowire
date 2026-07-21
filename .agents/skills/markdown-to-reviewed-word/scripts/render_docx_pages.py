#!/usr/bin/env python3
"""DOCXを専用LibreOffice profileでPDF化し、全pageをPNGへ変換する。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

MACOS_JAPANESE_FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")


def render_docx_pages(
    input_path: Path,
    output_dir: Path,
    *,
    dpi: int = 150,
    font_files: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """DOCXの全pageをPNGへ変換する。

    Args:
        input_path: 入力DOCX。
        output_dir: QA用PNGの出力directory。
        dpi: PNG rasterize解像度。既定150 dpi。
        font_files: LibreOfficeの一時profileへ供給するfont file。

    Returns:
        page順に並べたPNG pathのtuple。

    Raises:
        FileNotFoundError: DOCX、soffice、pdftoppmが存在しない場合。
        RuntimeError: PDFまたはPNG変換に失敗した場合。
        ValueError: dpiが正でない場合。
    """
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    soffice = shutil.which("soffice")
    pdftoppm = shutil.which("pdftoppm")
    if soffice is None:
        raise FileNotFoundError("soffice was not found in PATH")
    if pdftoppm is None:
        raise FileNotFoundError("pdftoppm was not found in PATH")

    output_dir.mkdir(parents=True, exist_ok=True)
    if tuple(output_dir.glob("page-*.png")):
        raise FileExistsError("output directory already contains page-*.png")
    with tempfile.TemporaryDirectory(prefix="chronowire-docx-") as profile_directory:
        profile_path = Path(profile_directory).resolve()
        profile_uri = profile_path.as_uri()
        supplied_fonts = font_files
        if not supplied_fonts and MACOS_JAPANESE_FONT.exists():
            supplied_fonts = (MACOS_JAPANESE_FONT,)
        if supplied_fonts:
            font_directory = profile_path / "user" / "fonts"
            font_directory.mkdir(parents=True)
            for font_file in supplied_fonts:
                if not font_file.exists():
                    raise FileNotFoundError(font_file)
                # bundled LibreOfficeはmacOSのsystem fontを自動発見しないため、
                # QA時だけ専用profileへcopyして日本語glyphの欠落を防ぐ。
                # system fontのimmutable flagは一時領域へ適用できないため、内容だけをcopyする。
                shutil.copyfile(font_file, font_directory / font_file.name)
        environment = os.environ.copy()
        environment["HOME"] = str(profile_path)
        environment["XDG_CACHE_HOME"] = str(profile_path / "cache")
        conversion = subprocess.run(
            [
                soffice,
                "--headless",
                f"-env:UserInstallation={profile_uri}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(input_path.resolve()),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    pdf_path = output_dir / f"{input_path.stem}.pdf"
    if conversion.returncode != 0 or not pdf_path.exists():
        raise RuntimeError(
            "LibreOffice conversion failed:\n"
            f"stdout: {conversion.stdout}\nstderr: {conversion.stderr}"
        )

    prefix = output_dir / "page"
    rasterize = subprocess.run(
        [pdftoppm, "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=False,
        capture_output=True,
        text=True,
    )
    pages = tuple(sorted(output_dir.glob("page-*.png")))
    if rasterize.returncode != 0 or not pages:
        raise RuntimeError(
            "PNG rendering failed:\n"
            f"stdout: {rasterize.stdout}\nstderr: {rasterize.stderr}"
        )
    return pages


def main() -> None:
    """CLI引数を読み、生成した全page pathを表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--font-file",
        action="append",
        default=[],
        type=Path,
        help="LibreOfficeの一時profileへcopyするfont file。複数指定可",
    )
    arguments = parser.parse_args()
    for page in render_docx_pages(
        arguments.docx,
        arguments.output_dir,
        dpi=arguments.dpi,
        font_files=tuple(arguments.font_file),
    ):
        print(page)


if __name__ == "__main__":
    main()
