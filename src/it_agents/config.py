"""Central configuration: paths, Azure OpenAI model factories, feature flags."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent.parent
CORPUS_DIR = Path(os.getenv("NFS_CORPUS_DIR", PACKAGE_DIR / "corpus"))
FIXTURES_DIR = Path(os.getenv("NFS_FIXTURES_DIR", PROJECT_DIR / "evals" / "fixtures"))
DATA_DIR = Path(os.getenv("NFS_DATA_DIR", PROJECT_DIR / ".data"))
CHROMA_DIR = DATA_DIR / "chroma"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
RUNS_DIR = DATA_DIR / "runs"
AUDIT_LOG = DATA_DIR / "audit_log.jsonl"
VENDOR_REGISTRY_FILE = PACKAGE_DIR / "data" / "vendor_registry.json"

COLLECTION_NAME = "nfs_corpus"

# Guardrail limits
MAX_TOOL_CALLS_PER_AGENT = int(os.getenv("NFS_MAX_TOOL_CALLS_PER_AGENT", "14"))
AGENT_RECURSION_LIMIT = int(os.getenv("NFS_AGENT_RECURSION_LIMIT", "150"))


@lru_cache
def get_llm(temperature: float = 0.0):
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=temperature,
        max_retries=3,
        # Streaming makes the timeout apply per chunk: long structured outputs complete,
        # while genuinely hung Azure requests fail fast and are retried.
        streaming=True,
        stream_usage=True,
        timeout=60,
        # Specialist outputs are ~2-3k tokens; the cap stops rare degenerate loops early so the retry kicks in.
        max_tokens=8000,
    )


@lru_cache
def get_embeddings():
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_EMBEDDING_ENDPOINT"],
        api_key=os.environ["AZURE_EMBEDDING_API_KEY"],
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        api_version=os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15"),
    )
