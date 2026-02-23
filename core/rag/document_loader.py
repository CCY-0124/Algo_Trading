"""
document_loader.py

Scans RAG_input: one folder per author. Under each author, loads post .txt files
(from "post" subfolder or directly in author folder) and from "scripts" subfolder:
.py files, .ipynb (Jupyter/Colab) notebooks. Image files are ignored.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional

from config.rag_config import (
    get_rag_input_dir,
    POST_FOLDER,
    SCRIPTS_FOLDER,
    EXPECTED_AUTHOR_COUNT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TEXT_EXTS = {".txt"}
# Scripts folder: .py, .ipynb, .json, and no extension (Colab may download with no type)
SCRIPT_EXTS = {".py", ".ipynb", ".json", ""}


def _read_text_file(path: Path, encoding: str = "utf-8") -> str:
    """Read text file; fallback to utf-8 with errors replaced."""
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return path.read_text(encoding=encoding, errors="replace")


def _is_notebook_json(obj: dict) -> bool:
    """Return True if the dict looks like a Jupyter/Colab notebook (has cells)."""
    return isinstance(obj.get("cells"), list)


def _read_ipynb(path: Path) -> str:
    """
    Read Jupyter/Colab notebook and return plain text from all cells.
    Works for .ipynb and for .json that Colab may use when downloaded from web.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        nb = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Not valid JSON: {e}") from e
    if not _is_notebook_json(nb):
        raise ValueError("JSON does not look like a notebook (no cells)")
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            parts.append("".join(src))
        else:
            parts.append(str(src))
    return "\n\n".join(p for p in parts if p.strip())


def _segment_text(text: str, max_segment: int = 2000) -> List[str]:
    """
    Split text into segments at natural boundaries (paragraph, then line).
    Segments longer than max_segment are split by character to avoid huge blocks.
    """
    segments = []
    for para in re.split(r"\n\n+", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_segment:
            segments.append(para)
        else:
            for line in para.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if len(line) <= max_segment:
                    segments.append(line)
                else:
                    for i in range(0, len(line), max_segment):
                        segments.append(line[i : i + max_segment])
    return segments


def _chunk_by_segments(
    segments: List[str],
    chunk_size: int,
    chunk_overlap: int,
    join_char: str = "\n\n",
) -> List[str]:
    """Merge segments into chunks of ~chunk_size, with overlap between consecutive chunks."""
    if not segments:
        return []
    chunks = []
    current = []
    current_len = 0
    for seg in segments:
        current.append(seg)
        current_len += len(seg) + (len(join_char) if len(current) > 1 else 0)
        if current_len >= chunk_size:
            chunk_text = join_char.join(current)
            chunks.append(chunk_text)
            overlap_parts = []
            overlap_len = 0
            for s in reversed(current):
                overlap_parts.insert(0, s)
                overlap_len += len(s) + (len(join_char) if len(overlap_parts) > 0 else 0)
                if overlap_len >= chunk_overlap and len(overlap_parts) < len(current):
                    break
            current = overlap_parts
            current_len = overlap_len
    if current:
        chunks.append(join_char.join(current))
    return chunks


# Skip system/hidden files that match no-extension (e.g. .DS_Store)
_SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def _collect_files(author_dir: Path, subfolder: Optional[str], exts: set, recursive: bool = False) -> List[Path]:
    """
    Collect files with given extensions from author_dir or author_dir/subfolder.
    Use '' in exts for no extension. If recursive is True and subfolder is set, scan subfolders too.
    Skips system files like .DS_Store and Thumbs.db.
    """
    def matches(p: Path) -> bool:
        if not p.is_file():
            return False
        if p.name in _SKIP_NAMES or p.name.startswith("."):
            return False
        suf = p.suffix.lower()
        return suf in exts
    if subfolder:
        dir_path = author_dir / subfolder
        if not dir_path.is_dir():
            return []
        if recursive:
            return sorted([p for p in dir_path.rglob("*") if matches(p)])
        return sorted([p for p in dir_path.iterdir() if matches(p)])
    return sorted([p for p in author_dir.iterdir() if matches(p)])


def discover_posts(rag_input_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Discover all documents under RAG_input. Each author has one folder. Under each
    author we load: (1) .txt from "post" subfolder if present, else .txt in author
    dir; (2) from "scripts" subfolder: .py (script) and .ipynb (notebook, Jupyter/Colab).
    Image files are skipped.

    :param rag_input_dir: Override RAG input root (default from config)
    :return: List of dicts: author, path, text, doc_type (post/script), metadata
    """
    root = Path(rag_input_dir) if rag_input_dir else get_rag_input_dir()
    if not root.is_dir():
        logging.warning("RAG input dir does not exist: %s", root)
        return []

    entries = []
    author_folders = sorted([d for d in root.iterdir() if d.is_dir()])
    if EXPECTED_AUTHOR_COUNT and len(author_folders) != EXPECTED_AUTHOR_COUNT:
        logging.warning(
            "RAG_input: expected %s author folder(s), found %s: %s",
            EXPECTED_AUTHOR_COUNT,
            len(author_folders),
            [d.name for d in author_folders],
        )

    for author_dir in author_folders:
        author_name = author_dir.name

        post_dir = author_dir / POST_FOLDER
        post_paths = _collect_files(author_dir, POST_FOLDER, TEXT_EXTS, recursive=True) if post_dir.is_dir() else _collect_files(author_dir, None, TEXT_EXTS)

        for path in post_paths:
            try:
                text = _read_text_file(path)
            except Exception as e:
                logging.warning("Skip %s: %s", path, e)
                continue
            rel_path = path.relative_to(root)
            entries.append({
                "author": author_name,
                "path": path,
                "text": text,
                "doc_type": "post",
                "metadata": {
                    "author": author_name,
                    "doc_type": "post",
                    "file_name": path.name,
                    "post_basename": path.stem,
                    "source_file": f"{author_name}|{rel_path.as_posix()}",
                },
            })

        scripts_dir = author_dir / SCRIPTS_FOLDER
        if scripts_dir.is_dir():
            for path in _collect_files(author_dir, SCRIPTS_FOLDER, SCRIPT_EXTS, recursive=True):
                try:
                    suf = path.suffix.lower()
                    if suf == ".py":
                        text = _read_text_file(path)
                        doc_type = "script"
                    else:
                        text = _read_ipynb(path)
                        doc_type = "notebook"
                except Exception as e:
                    logging.warning("Skip %s: %s", path, e)
                    continue
                if not text.strip():
                    logging.warning("Skip %s: empty content", path)
                    continue
                rel_path = path.relative_to(root)
                entries.append({
                    "author": author_name,
                    "path": path,
                    "text": text,
                    "doc_type": doc_type,
                    "metadata": {
                        "author": author_name,
                        "doc_type": doc_type,
                        "file_name": path.name,
                        "post_basename": path.stem,
                        "source_file": f"{author_name}|{rel_path.as_posix()}",
                    },
                })

    return entries


def load_rag_documents(
    rag_input_dir: Optional[Path] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    use_boundaries: bool = True,
) -> Iterator[Dict[str, Any]]:
    """
    Load documents from RAG_input for import. Each item is one or more chunks
    with metadata (author, doc_type, post_basename).

    Slicing: when use_boundaries is True (default), text is split at paragraph
    then line boundaries, then segments are merged until ~chunk_size so chunks
    do not cut mid sentence. Overlap is applied by reusing the last segment(s)
    at the start of the next chunk. When use_boundaries is False, fixed
    character windows are used (may cut mid word).

    :param rag_input_dir: Override RAG input root
    :param chunk_size: Target size in characters per chunk
    :param chunk_overlap: Overlap in characters between consecutive chunks
    :param use_boundaries: If True, chunk at paragraph/line boundaries
    :yield: Dict with keys: text, metadata (author, doc_type, post_basename, chunk_id)
    """
    entries = discover_posts(rag_input_dir)
    for entry in entries:
        text = entry["text"]
        meta = dict(entry["metadata"])
        if len(text) <= chunk_size:
            meta_single = {**meta, "chunk_id": 0}
            yield {"text": text, "metadata": meta_single}
            continue
        if use_boundaries:
            segments = _segment_text(text, max_segment=chunk_size)
            chunk_texts = _chunk_by_segments(segments, chunk_size, chunk_overlap)
        else:
            chunk_texts = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_texts.append(text[start:end])
                start = end - chunk_overlap if end < len(text) else len(text)
        for chunk_id, chunk in enumerate(chunk_texts):
            if not chunk.strip():
                continue
            meta_chunk = {**meta, "chunk_id": chunk_id}
            yield {"text": chunk, "metadata": meta_chunk}
