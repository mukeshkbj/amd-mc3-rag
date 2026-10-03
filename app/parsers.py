"""Per-type corpus parsers. Stdlib-first: docx/xlsx are just zipped XML.

Every function returns a list of text segments (paragraph/row-sized), or raises.
Callers catch per-file so one unreadable file never aborts the walk.
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

DEAD_NAME = re.compile(r"withdrawn|superseded|deprecated|obsolete", re.I)
# Self-declared status banners appear in the header region; later mentions
# ("r2 supersedes r1", "revision 1 is withdrawn") are history, not status.
DEAD_HEAD = re.compile(
    r"withdrawn|superseded by|deprecated|obsolete|do not use|do not design",
    re.I)
TEXT_EXT = {".txt", ".log", ".py", ".md", ".json", ".yaml", ".yml", ".ini", ".cfg"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


class Unreadable(Exception):
    """Permission denied, encrypted, unknown type, corrupt — skip this file."""


def is_dead_name(name: str) -> bool:
    return bool(DEAD_NAME.search(name))


def looks_dead(text: str) -> bool:
    return bool(DEAD_HEAD.search(text[:400]))


def parse_text_file(path: Path) -> list[str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return text.splitlines()


def parse_csv_file(path: Path) -> list[str]:
    rows = list(csv.reader(io.StringIO(
        path.read_text(encoding="utf-8", errors="replace"))))
    if not rows:
        return []
    header, out = rows[0], []
    for r in rows[1:]:
        cells = [f"{header[i].strip()}: {c.strip()}" for i, c in enumerate(r)
                 if i < len(header) and header[i].strip() and c.strip()]
        if cells:
            out.append(" | ".join(cells))
    return out


def _docx_segments(zf: zipfile.ZipFile) -> list[str]:
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    root = ET.fromstring(zf.read("word/document.xml"))
    out = []
    for para in root.iter(f"{W}p"):
        text = "".join(t.text or "" for t in para.iter(f"{W}t"))
        if text.strip():
            out.append(text.strip())
    for tbl in root.iter(f"{W}tbl"):
        for tr in tbl.iter(f"{W}tr"):
            cells = []
            for tc in tr.iter(f"{W}tc"):
                cell = "".join(t.text or "" for t in tc.iter(f"{W}t")).strip()
                cells.append(cell)
            if any(cells):
                out.append(" | ".join(cells))
    return out


def _xlsx_segments(zf: zipfile.ZipFile) -> list[str]:
    NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    shared = []
    if "xl/sharedStrings.xml" in zf.namelist():
        sroot = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        for si in sroot.iter(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    sheet_names = {}
    if "xl/workbook.xml" in zf.namelist():
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        for i, sh in enumerate(wb.iter(f"{NS}sheet"), 1):
            sheet_names[i] = sh.get("name", f"sheet{i}")
    out = []
    for name in sorted(n for n in zf.namelist()
                       if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)):
        num = int(re.search(r"(\d+)", Path(name).stem).group(1))
        sname = sheet_names.get(num, Path(name).stem)
        sroot = ET.fromstring(zf.read(name))
        rows = []
        for row in sroot.iter(f"{NS}row"):
            vals = []
            for c in row.iter(f"{NS}c"):
                if c.get("t") == "s":
                    v = c.find(f"{NS}v")
                    vals.append(shared[int(v.text)] if v is not None
                                and v.text is not None else "")
                elif c.get("t") == "inlineStr":
                    vals.append("".join(t.text or "" for t in c.iter(f"{NS}t")))
                else:
                    v = c.find(f"{NS}v")
                    vals.append(v.text if v is not None and v.text else "")
            if any(v.strip() for v in vals):
                rows.append(vals)
        if not rows:
            continue
        header = rows[0]
        for r in rows[1:]:
            cells = [f"{header[i]}: {v}" for i, v in enumerate(r)
                     if i < len(header) and header[i] and str(v).strip()]
            if cells:
                out.append(f"[{sname}] " + " | ".join(cells))
    return out


def parse_office(path: Path) -> list[str]:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise Unreadable(f"not a zip: {e}")
    if path.suffix.lower() == ".docx":
        return _docx_segments(zf)
    return _xlsx_segments(zf)


def parse_pdf(path: Path) -> list[str]:
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(path)
    except Exception as e:
        raise Unreadable(f"pdf open failed: {e}")
    try:
        if doc.needs_pass or doc.is_encrypted:
            raise Unreadable("encrypted pdf")
        out = []
        for page in doc:
            for line in page.get_text().splitlines():
                if line.strip():
                    out.append(line.strip())
        return out
    finally:
        doc.close()


def parse_file(path: Path, engine=None) -> tuple[list[str], bool]:
    """-> (segments, superseded). Raises Unreadable for anything we can't read.

    engine is required only for image files (VLM transcription).
    """
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        if engine is None:
            raise Unreadable("image needs engine")
        return [engine.transcribe_image(str(path))], is_dead_name(path.name)
    if ext == ".pdf":
        segs = parse_pdf(path)
    elif ext in (".docx", ".xlsx"):
        segs = parse_office(path)
    elif ext == ".csv":
        segs = parse_csv_file(path)
    elif ext in TEXT_EXT:
        segs = parse_text_file(path)
    else:
        raise Unreadable(f"unknown type {ext or path.name}")
    if not segs:
        raise Unreadable("no text extracted")
    dead = is_dead_name(path.name) or looks_dead(" ".join(segs[:20]))
    return segs, dead
