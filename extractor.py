"""
extractor.py
- extract_text_from_pdf(pdf_path): returns list of pages' text
- extract_chunks_from_termsheet(termsheet_pdf, chunk_size=2000, overlap=200)
  returns list of dicts: { "chunk": str, "source": "termsheet", "page": int, "folder": folder_name }
"""

import pdfplumber
from typing import List, Dict

def extract_text_from_pdf(path: str) -> List[str]:
    """Return list of page texts (index 0 == page 1). Uses pdfplumber."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text = p.extract_text() or ""
            pages.append(text)
    return pages

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Naive chunking by characters with overlap."""
    if not text:
        return []
    chunks = []
    step = chunk_size - overlap
    i = 0
    while i < len(text):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk.strip())
        i += step
    return chunks

def extract_chunks_from_termsheet(termsheet_pdf: str,chunk_size: int = 500, overlap: int = 200,folder_name: str = None) -> List[Dict]:
    """
    Extracts text from a single Term Sheet PDF, chunks per page,
    and returns list of chunk dicts.
    Each dict: {chunk, source="termsheet", page, chunk_index, folder}
    """
    all_chunks = []

    pages = extract_text_from_pdf(termsheet_pdf)
    for idx, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for c_idx, chunk in enumerate(page_chunks, start=1):
            all_chunks.append({
                "chunk": chunk,
                "source": "termsheet",
                "page": idx,
                "chunk_index": c_idx,
                "folder": folder_name
            })

    return all_chunks
