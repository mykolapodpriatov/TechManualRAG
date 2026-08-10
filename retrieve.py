"""Local semantic search over ingested PDF chunks.

Chunks produced by :mod:`ingest` are embedded with a local BGE-small
``sentence-transformers`` model and stored in an embedded (serverless) Qdrant
collection on disk — no Qdrant server and no network access is required at
query time.

Both :func:`build_index` and :func:`search` accept an optional ``embedder``
override so callers (and the test suite) can swap in a tiny stub instead of
loading the real BGE model, the same spirit as ``test_ingest.py`` staying
independent of the heavy stack.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingest import PageChunks, chunk_id

DEFAULT_COLLECTION = "manuals"
DEFAULT_QDRANT_PATH = "data/qdrant"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# A text embedder: takes N texts, returns N equal-length dense vectors.
Embedder = Callable[[list[str]], list[list[float]]]

# Arbitrary but fixed namespace used to turn chunk ids into Qdrant-compatible
# UUIDs (Qdrant point ids must be an unsigned int or a UUID).
_POINT_ID_NAMESPACE = uuid.UUID("c9f1b2a4-6e3d-4a3b-9c8e-2f6d1a7b5e40")

_model_cache: dict[str, object] = {}


def _bge_embedder(texts: list[str]) -> list[list[float]]:
    """Default embedder: a local ``sentence-transformers`` BGE-small model.

    The model is loaded once per process and cached. Importing
    ``sentence_transformers`` happens lazily, on first use, so code paths
    that always inject their own ``embedder`` (e.g. the test suite) never
    need it installed.
    """
    model = _model_cache.get("model")
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(DEFAULT_MODEL_NAME)
        _model_cache["model"] = model
    vectors = model.encode(texts, normalize_embeddings=True)  # type: ignore[attr-defined]
    return [vector.tolist() for vector in vectors]


def embed_chunks(
    texts: list[str], *, embedder: Embedder | None = None
) -> list[list[float]]:
    """Embed ``texts`` into dense vectors.

    Args:
        texts: The strings to embed.
        embedder: Override for the embedding backend, mainly for tests. When
            omitted, a local BGE-small ``sentence-transformers`` model is
            used (downloaded/cached on first use).

    Returns:
        One embedding vector per input text, in the same order. Empty when
        ``texts`` is empty (the embedder is never invoked in that case).
    """
    if not texts:
        return []
    embed = embedder or _bge_embedder
    return embed(texts)


@dataclass(frozen=True)
class SearchResult:
    """One ranked search hit."""

    chunk_id: str
    page_no: int
    text: str
    score: float


def _point_id(cid: str) -> str:
    """Map a stable :func:`ingest.chunk_id` string to a Qdrant point id.

    Deterministic, so re-indexing the same document updates its existing
    points instead of creating duplicates.
    """
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, cid))


def _open_client(path: str) -> QdrantClient:
    """Open a local (embedded) Qdrant client backed by ``path``."""
    return QdrantClient(path=path)


def build_index(
    page_chunks: list[PageChunks],
    collection_name: str,
    path: str = DEFAULT_QDRANT_PATH,
    *,
    embedder: Embedder | None = None,
) -> None:
    """Embed every chunk in ``page_chunks`` and upsert it into Qdrant.

    Uses an on-disk, embedded Qdrant collection at ``path`` — no external
    Qdrant service is required. Points are keyed by :func:`ingest.chunk_id`,
    so re-running this for the same document updates its vectors in place
    rather than duplicating them.

    Args:
        page_chunks: Per-page chunks, as produced by :func:`ingest.chunk_pages`.
        collection_name: Qdrant collection to create (if missing) and upsert into.
        path: Directory for the local Qdrant store.
        embedder: Optional embedding backend override; see :func:`embed_chunks`.
    """
    records = [
        (chunk_id(pc.page_no, index), pc.page_no, text)
        for pc in page_chunks
        for index, text in enumerate(pc.chunks)
    ]
    if not records:
        return

    texts = [text for _, _, text in records]
    vectors = embed_chunks(texts, embedder=embedder)

    client = _open_client(path)
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=len(vectors[0]), distance=qmodels.Distance.COSINE
                ),
            )
        points = [
            qmodels.PointStruct(
                id=_point_id(cid),
                vector=vector,
                payload={"chunk_id": cid, "page_no": page_no, "text": text},
            )
            for (cid, page_no, text), vector in zip(records, vectors, strict=True)
        ]
        client.upsert(collection_name=collection_name, points=points)
    finally:
        client.close()


def search(
    query: str,
    collection_name: str,
    top_k: int = 5,
    path: str = DEFAULT_QDRANT_PATH,
    *,
    embedder: Embedder | None = None,
) -> list[SearchResult]:
    """Return the ``top_k`` chunks most relevant to ``query``.

    Args:
        query: Free-text search query.
        collection_name: Qdrant collection to search, as passed to :func:`build_index`.
        top_k: Maximum number of results to return.
        path: Directory of the local Qdrant store, as passed to :func:`build_index`.
        embedder: Optional embedding backend override; see :func:`embed_chunks`.

    Returns:
        Ranked :class:`SearchResult` items, best match first. Empty if the
        collection doesn't exist yet (nothing has been indexed).
    """
    client = _open_client(path)
    try:
        if not client.collection_exists(collection_name):
            return []

        [query_vector] = embed_chunks([query], embedder=embedder)
        hits = client.query_points(
            collection_name=collection_name, query=query_vector, limit=top_k
        ).points
        return [
            SearchResult(
                chunk_id=hit.payload["chunk_id"],
                page_no=hit.payload["page_no"],
                text=hit.payload["text"],
                score=hit.score,
            )
            for hit in hits
            if hit.payload is not None
        ]
    finally:
        client.close()
