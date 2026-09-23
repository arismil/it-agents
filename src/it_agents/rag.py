"""RAG over the NFS Knowledge Corpus: PDF ingestion into Chroma + filtered retrieval."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import CHROMA_DIR, CHUNKS_FILE, COLLECTION_NAME, CORPUS_DIR, FIXTURES_DIR, VENDOR_REGISTRY_FILE, get_embeddings

log = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"\b(?:POL|PROP|SCH|SEC)-[A-Z0-9]+(?:-[A-Z0-9]+)+\b")


@lru_cache
def load_registry() -> list[dict]:
    return json.loads(VENDOR_REGISTRY_FILE.read_text())["vendors"]


def resolve_vendor(name: str) -> dict | None:
    """Match a vendor name/alias (case-insensitive, substring tolerant) to the registry."""
    n = name.strip().lower()
    for v in load_registry():
        names = [v["vendor_name"].lower()] + [a.lower() for a in v["aliases"]]
        if n in names or any(x in n or n in x for x in names if len(x) > 4 and len(n) > 4):
            return v
    return None


def _vendor_for_file(filename: str) -> str | None:
    for v in load_registry():
        for alias in sorted([v["vendor_name"], *v["aliases"]], key=len, reverse=True):
            if filename.lower().startswith(alias.lower()):
                return v["vendor_name"]
    return None


def _clean(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"[ \t]+", " ", text).strip()


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    pages = [_clean(p.extract_text() or "") for p in reader.pages]
    m = _DOC_ID_RE.search(pages[0] if pages else "")
    doc_id = m.group(0) if m else "HIST-" + re.sub(r"[^A-Z0-9]+", "-", path.stem.upper()).strip("-").replace("-ASSESSMENT", "")
    is_policy = path.name.startswith("NFS")
    meta = {
        "doc_id": doc_id,
        "title": path.stem,
        "doc_type": "policy" if is_policy else "vendor",
        "vendor": "" if is_policy else (_vendor_for_file(path.name) or ""),
        "fixture": False,
    }
    return [Document(page_content=t, metadata={**meta, "page": i + 1}) for i, t in enumerate(pages) if t]


def _load_markdown(path: Path) -> list[Document]:
    """Fixture format: 'key: value' header lines, a '---' line, then the body."""
    raw = path.read_text()
    header, _, body = raw.partition("\n---\n")
    meta = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    return [Document(page_content=_clean(body), metadata={
        "doc_id": meta["doc_id"], "title": meta.get("title", path.stem), "doc_type": meta.get("doc_type", "vendor"),
        "vendor": meta.get("vendor", ""), "fixture": True, "page": 1,
    })]


def load_documents(include_fixtures: bool = True) -> list[Document]:
    docs = []
    for p in sorted(CORPUS_DIR.glob("*.pdf")):
        docs += _load_pdf(p)
    if include_fixtures and FIXTURES_DIR.exists():
        for p in sorted(FIXTURES_DIR.glob("*.md")):
            docs += _load_markdown(p)
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d.page_content)):
            meta = {**d.metadata, "chunk_id": f"{d.metadata['doc_id']}:p{d.metadata['page']}:c{i}"}
            chunks.append(Document(page_content=piece, metadata=meta))
    return chunks


def get_store():
    from langchain_chroma import Chroma

    return Chroma(collection_name=COLLECTION_NAME, embedding_function=get_embeddings(),
                  persist_directory=str(CHROMA_DIR))


def ingest(reset: bool = True) -> dict:
    """(Re)build the vector index and the chunk registry used for citation verification."""
    chunks = chunk_documents(load_documents())
    store = get_store()
    if reset:
        store.reset_collection()
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps({**c.metadata, "text": c.page_content}) + "\n")
    docs = sorted({(c.metadata["doc_id"], c.metadata["title"]) for c in chunks})
    log.info("Ingested %d chunks from %d documents", len(chunks), len(docs))
    return {"chunks": len(chunks), "documents": [{"doc_id": d, "title": t} for d, t in docs]}


def index_ready() -> bool:
    return CHUNKS_FILE.exists() and CHUNKS_FILE.stat().st_size > 0


def search(query: str, k: int = 5, doc_type: str | None = None, vendor: str | None = None) -> list[Document]:
    clauses = []
    if doc_type:
        clauses.append({"doc_type": doc_type})
    if vendor is not None:
        clauses.append({"vendor": vendor})
    where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
    return get_store().similarity_search(query, k=k, filter=where)
