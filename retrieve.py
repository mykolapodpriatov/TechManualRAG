"""Local semantic search over ingested PDF chunks.

Chunks produced by :mod:`ingest` are embedded with a local BGE-small
``sentence-transformers`` model and stored in an embedded (serverless) Qdrant
collection on disk — no Qdrant server and no network access is required at
query time.

Both :func:`build_index` and :func:`search` accept an optional ``embedder``
override so callers (and the test suite) can swap in a tiny stub instead of
loading the real BGE model, the same spirit as ``test_ingest.py`` staying
independent of the heavy stack.

It also doubles as an offline CLI::

    python retrieve.py "torque spec" --collection manuals
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingest import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    Page,
    PageChunks,
    chunk_id,
    chunk_pages,
)

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


def index_pages(
    pages: list[Page],
    collection_name: str,
    path: str = DEFAULT_QDRANT_PATH,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    words: bool = False,
    embedder: Embedder | None = None,
) -> int:
    """Chunk ``pages`` and index them, returning how many chunks were stored.

    This is the one path from extracted pages to a searchable collection, so
    the Streamlit UI and the ``ingest.py --index`` CLI produce identical chunk
    ids for the same document and chunking settings. Re-indexing a document
    updates its points in place, because :func:`build_index` keys them on
    :func:`ingest.chunk_id`.

    Args:
        pages: Extracted pages, as produced by :func:`ingest.extract_pages`.
        collection_name: Qdrant collection to create (if missing) and upsert into.
        path: Directory for the local Qdrant store.
        chunk_size: Chunk window, in characters (or words when ``words``).
        overlap: Overlap between neighboring chunks.
        words: Chunk on whole-word boundaries instead of characters.
        embedder: Optional embedding backend override; see :func:`embed_chunks`.

    Returns:
        The number of chunks embedded and upserted. Zero when ``pages`` holds
        no extractable text, in which case nothing is written.

    Raises:
        ValueError: If ``chunk_size`` / ``overlap`` are invalid, as raised by
            :func:`ingest.chunk_pages`.
    """
    page_chunks = chunk_pages(pages, chunk_size, overlap, words=words)
    total = sum(len(pc.chunks) for pc in page_chunks)
    if total == 0:
        return 0
    build_index(page_chunks, collection_name, path=path, embedder=embedder)
    return total


def _page_range_filter(pages: tuple[int, int] | None) -> qmodels.Filter | None:
    """Build an inclusive ``page_no`` payload filter, or ``None`` if unset.

    Raises:
        ValueError: If the range is inverted, empty, or not 1-based.
    """
    if pages is None:
        return None
    lo, hi = pages
    if lo < 1 or hi < lo:
        raise ValueError(f"invalid pages range {pages!r}; expected 1 <= start <= end")
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="page_no",
                range=qmodels.Range(gte=lo, lte=hi),
            )
        ]
    )


def search(
    query: str,
    collection_name: str,
    top_k: int = 5,
    path: str = DEFAULT_QDRANT_PATH,
    *,
    pages: tuple[int, int] | None = None,
    min_score: float | None = None,
    embedder: Embedder | None = None,
) -> list[SearchResult]:
    """Return the ``top_k`` chunks most relevant to ``query``.

    Args:
        query: Free-text search query.
        collection_name: Qdrant collection to search, as passed to :func:`build_index`.
        top_k: Maximum number of results to return.
        path: Directory of the local Qdrant store, as passed to :func:`build_index`.
        pages: Optional inclusive 1-based ``(start, end)`` page range. When set,
            only chunks whose ``page_no`` falls inside the range are returned.
        min_score: Optional cosine-score floor in ``[0, 1]``. Hits below it
            are dropped after the Qdrant query. ``None`` keeps every hit.
        embedder: Optional embedding backend override; see :func:`embed_chunks`.

    Returns:
        Ranked :class:`SearchResult` items, best match first. Empty if the
        collection doesn't exist yet (nothing has been indexed).

    Raises:
        ValueError: If ``pages`` is inverted or empty (``end < start``) or
            not 1-based, or if ``min_score`` is outside ``[0, 1]``.
    """
    if min_score is not None and not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0, 1], got {min_score!r}")

    query_filter = _page_range_filter(pages)

    client = _open_client(path)
    try:
        if not client.collection_exists(collection_name):
            return []

        [query_vector] = embed_chunks([query], embedder=embedder)
        hits = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        ).points
        results = [
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

    if min_score is None:
        return results
    return [result for result in results if result.score >= min_score]


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #

_EXIT_USAGE = 2
_SNIPPET_CHARS = 80


def _collection_exists(collection_name: str, path: str) -> bool:
    """Return whether ``collection_name`` exists in the local Qdrant store."""
    client = _open_client(path)
    try:
        return client.collection_exists(collection_name)
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the ``retrieve`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="retrieve",
        description="Search a local Qdrant collection of ingested PDF chunks.",
    )
    parser.add_argument("query", help="Free-text search query.")
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        metavar="NAME",
        help=f"Qdrant collection to search (default: {DEFAULT_COLLECTION!r}).",
    )
    parser.add_argument(
        "--qdrant-path",
        default=DEFAULT_QDRANT_PATH,
        metavar="PATH",
        help=f"On-disk path for the local Qdrant store (default: {DEFAULT_QDRANT_PATH!r}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of results to return (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit a JSON array of {id, page, text, score} objects to stdout.",
    )
    return parser


def _fail(message: str) -> int:
    """Print an error to stderr and return the usage exit code."""
    print(f"error: {message}", file=sys.stderr)
    return _EXIT_USAGE


def main(argv: Sequence[str] | None = None, *, embedder: Embedder | None = None) -> int:
    """Run the CLI. Returns the process exit code (0 on success)."""
    args = build_parser().parse_args(argv)

    if not args.query.strip():
        return _fail("query is empty")

    if not _collection_exists(args.collection, args.qdrant_path):
        return _fail(
            f"collection {args.collection!r} does not exist yet. "
            "Index a manual first: python ingest.py manual.pdf --index"
        )

    results = search(
        args.query,
        args.collection,
        top_k=args.top_k,
        path=args.qdrant_path,
        embedder=embedder,
    )

    if args.as_json:
        payload = [
            {
                "id": result.chunk_id,
                "page": result.page_no,
                "text": result.text,
                "score": result.score,
            }
            for result in results
        ]
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for index, result in enumerate(results, start=1):
            snippet = " ".join(result.text[:_SNIPPET_CHARS].split())
            print(f"{index}. page {result.page_no}  score {result.score:.3f}")
            print(f"   {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
