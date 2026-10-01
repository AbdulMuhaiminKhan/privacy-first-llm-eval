"""Render data/sample/handbook_source.txt into a multi-page PDF (one section per page).

The committed handbook.pdf is the evaluation corpus; re-run this only if you edit the source.
Requires the dev dependency `reportlab`.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sample" / "handbook_source.txt"
OUT = ROOT / "data" / "sample" / "handbook.pdf"
PAGE_MARKER = "=== PAGE ==="


def main() -> None:
    styles = getSampleStyleSheet()
    pages = [p.strip() for p in SRC.read_text(encoding="utf-8").split(PAGE_MARKER) if p.strip()]
    story = []
    for i, page in enumerate(pages):
        paragraphs = [p.strip() for p in page.split("\n\n") if p.strip()]
        for j, para in enumerate(paragraphs):
            style = styles["Heading2"] if j == 0 else styles["BodyText"]
            story.append(Paragraph(para.replace("\n", " "), style))
            story.append(Spacer(1, 6))
        if i < len(pages) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(OUT), pagesize=A4, title="Quillfeather Handbook (fictional)").build(story)
    print(f"Wrote {OUT} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
