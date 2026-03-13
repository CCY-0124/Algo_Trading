"""
rag_config.py

RAG paths. Override via env: RAG_INPUT_DIR, RAG_CHROMA_DIR.
"""

import os
from pathlib import Path

# Project root (config's parent)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAG_INPUT_DIR = Path(os.environ.get("RAG_INPUT_DIR", _PROJECT_ROOT / "RAG_input"))
RAG_CHROMA_DIR = Path(os.environ.get("RAG_CHROMA_DIR", _PROJECT_ROOT / "chroma_db"))

# Document layout under each author folder
POST_FOLDER = "post"
SCRIPTS_FOLDER = "scripts"
# Set to a number to warn if author count differs; None to skip check
EXPECTED_AUTHOR_COUNT = None


def get_rag_input_dir() -> Path:
    return RAG_INPUT_DIR


def get_chroma_dir() -> Path:
    return RAG_CHROMA_DIR
