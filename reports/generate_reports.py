from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import REPORTS_DIR


PDF_TEXT = """
XAI Finance Platform - System Design
System architecture: FastAPI handles prediction APIs, Streamlit handles visualization, SQLite stores platform state.
Module responsibilities: data processing, model training, fairness evaluation, explainability, monitoring, and audit logging are separated by package.
Explainability methods: SHAP supports global and local attribution, LIME supports local surrogate explanations, and ELI5 provides readable summaries.
Fairness checks: demographic parity difference and equal opportunity difference are evaluated by sex and age group.
Monitoring: the platform tracks prediction latency, explanation latency, and SHAP versus LIME feature overlap.
""".strip()

DOCX_TEXT = """
XAI Finance Platform - Implementation Report
This report summarizes the system architecture, module responsibilities, explainability workflow, fairness checks, and monitoring strategy for the local prototype.
""".strip()


def escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def create_simple_pdf(path: Path, text: str) -> None:
    lines = [escape_pdf_text(line) for line in textwrap.wrap(text.replace("\n", " "), width=90)]
    stream_lines = ["BT", "/F1 12 Tf", "72 770 Td"]
    for index, line in enumerate(lines):
        if index:
            stream_lines.append("0 -16 Td")
        stream_lines.append(f"({line}) Tj")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines).encode("utf-8")
    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n")
    objects.append(f"4 0 obj << /Length {len(stream)} >> stream\n".encode("utf-8") + stream + b"\nendstream endobj\n")
    objects.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    header = b"%PDF-1.4\n"
    offsets = []
    body = b""
    for obj in objects:
        offsets.append(len(header) + len(body))
        body += obj
    xref_offset = len(header) + len(body)
    xref = [b"xref\n0 6\n0000000000 65535 f \n"]
    for offset in offsets:
        xref.append(f"{offset:010d} 00000 n \n".encode("utf-8"))
    trailer = b"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n" + str(xref_offset).encode("utf-8") + b"\n%%EOF"
    path.write_bytes(header + body + b"".join(xref) + trailer)


def create_simple_docx(path: Path, text: str) -> None:
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    paragraphs = "".join(f"<w:p><w:r><w:t>{line}</w:t></w:r></w:p>" for line in text.splitlines() if line.strip())
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>{paragraphs}<w:sectPr/></w:body>
</w:document>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)


if __name__ == "__main__":
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    create_simple_pdf(REPORTS_DIR / "system_design.pdf", PDF_TEXT)
    create_simple_docx(REPORTS_DIR / "implementation_report.docx", DOCX_TEXT)
