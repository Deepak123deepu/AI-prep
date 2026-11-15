"""ChromaDB equivalent of the Qdrant memory store.

This module connects to a Chroma server at localhost:8000. Start one locally
with ``chroma run --path .chroma --port 8000`` before calling its functions.
The public functions are async so they can be used like the Qdrant versions;
the synchronous Chroma HTTP calls run in worker threads.
"""

import asyncio
from typing import Any, Optional
from uuid import uuid4

import chromadb
from pydantic import BaseModel


client = chromadb.HttpClient(host="localhost", port=8000)
COLLECTION_NAME = "memories"
SIMILARITY_THRESHOLD = 0.1
SEARCH_LIMIT = 2


class EmbeddedMemory(BaseModel):
    """A memory record ready to store, including its precomputed embedding."""

    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    embedding: list[float]


class RetrievedMemory(BaseModel):
    """A memory read from Chroma; score is absent when no similarity search ran."""

    point_id: str
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    score: Optional[float] = None


def _get_collection_sync():
    """Get the existing collection; callers run this blocking operation in a thread."""
    return client.get_collection(name=COLLECTION_NAME)


def _build_where_filter(
    user_id: int,
    categories: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build a Chroma metadata filter for one user and optionally matching categories.

    Chroma stores categories as a metadata array. The ``$contains`` tests inside
    ``$or`` match a record if it has any one of the requested categories.
    """
    user_filter: dict[str, Any] = {"user_id": user_id}
    if not categories:
        return user_filter

    return {
        "$and": [
            user_filter,
            {
                "$or": [
                    {"categories": {"$contains": category}}
                    for category in categories
                ]
            },
        ]
    }


async def create_memory_collection() -> None:
    """Create the ``memories`` collection if needed, using inner-product distance.

    Inner product matches the distance metric used by the Qdrant version. Chroma
    determines the vector size from the first inserted embedding, rather than
    requiring a size when the collection is created. Metadata filtering is
    supported directly, so separate payload-index creation is unnecessary.
    """
    collection = await asyncio.to_thread(
        client.get_or_create_collection,
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "ip"},
    )
    metric = (collection.metadata or {}).get("hnsw:space", "l2")
    if metric != "ip":
        raise ValueError(
            f"Collection {COLLECTION_NAME!r} uses {metric!r} distance; "
            "recreate it with hnsw:space='ip' to match the Qdrant behavior."
        )


async def insert_memories(memories: list[EmbeddedMemory]) -> None:
    """Insert or replace memory records in Chroma.

    A fresh UUID is assigned to each record. The memory text is stored as the
    document, searchable fields are stored as metadata, and the supplied
    embedding is stored as the vector. ``upsert`` allows safe replacement if an
    ID is ever reused.
    """
    if not memories:
        return

    collection = await asyncio.to_thread(_get_collection_sync)
    await asyncio.to_thread(
        collection.upsert,
        ids=[uuid4().hex for _ in memories],
        embeddings=[memory.embedding for memory in memories],
        documents=[memory.memory_text for memory in memories],
        metadatas=[
            {
                "user_id": memory.user_id,
                "categories": memory.categories,
                "date": memory.date,
            }
            for memory in memories
        ],
    )


async def search_memories(
    search_vector: list[float],
    user_id: int,
    categories: Optional[list[str]] = None,
) -> list[RetrievedMemory]:
    """Return up to two similar memories belonging to a user.

    If categories are supplied, a record matches when it contains at least one
    requested category. Chroma returns inner-product *distance* (1 - similarity),
    so this function converts it back to a similarity score and keeps only
    results meeting the Qdrant version's 0.1 score threshold.
    """
    collection = await asyncio.to_thread(_get_collection_sync)
    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[search_vector],
        n_results=SEARCH_LIMIT,
        where=_build_where_filter(user_id, categories),
        include=["metadatas", "documents", "distances"],
    )

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    retrieved: list[RetrievedMemory] = []
    for point_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        score = 1.0 - float(distance)
        if score >= SIMILARITY_THRESHOLD:
            retrieved.append(
                convert_retrieved_record(
                    point_id=point_id,
                    metadata=metadata,
                    document=document,
                    score=score,
                )
            )
    return retrieved


async def delete_user_records(user_id: int) -> None:
    """Delete every memory owned by ``user_id`` without affecting other users."""
    collection = await asyncio.to_thread(_get_collection_sync)
    await asyncio.to_thread(collection.delete, where={"user_id": user_id})


async def delete_records(point_ids: list[str]) -> None:
    """Delete records by their Chroma IDs; an empty ID list is a no-op."""
    if not point_ids:
        return

    collection = await asyncio.to_thread(_get_collection_sync)
    await asyncio.to_thread(collection.delete, ids=point_ids)


async def fetch_all_user_records(user_id: int) -> list[RetrievedMemory]:
    """Fetch all of a user's memories without ranking them by similarity.

    Since this is an unranked fetch, returned records have ``score=None``.
    """
    collection = await asyncio.to_thread(_get_collection_sync)
    results = await asyncio.to_thread(
        collection.get,
        where={"user_id": user_id},
        include=["metadatas", "documents"],
    )
    return [
        convert_retrieved_record(
            point_id=point_id,
            metadata=metadata,
            document=document,
            score=None,
        )
        for point_id, metadata, document in zip(
            results["ids"], results["metadatas"], results["documents"]
        )
    ]


def convert_retrieved_record(
    point_id: str,
    metadata: dict[str, Any],
    document: str,
    score: Optional[float],
) -> RetrievedMemory:
    """Convert Chroma's separate ID, metadata, document, and score into a model."""
    return RetrievedMemory(
        point_id=point_id,
        user_id=metadata["user_id"],
        memory_text=document,
        categories=metadata["categories"],
        date=metadata["date"],
        score=score,
    )


async def get_all_categories(user_id: int) -> list[str]:
    """Return the sorted unique category names used by one user.

    Unlike Qdrant's facet operation, Chroma has no equivalent facet endpoint;
    this fetches the user's category metadata and deduplicates it in Python.
    """
    collection = await asyncio.to_thread(_get_collection_sync)
    results = await asyncio.to_thread(
        collection.get,
        where={"user_id": user_id},
        include=["metadatas"],
    )
    categories = {
        category
        for metadata in results["metadatas"]
        for category in metadata["categories"]
    }
    return sorted(categories)


def stringify_retrieved_point(retrieved_memory: RetrievedMemory) -> str:
    """Format a retrieved memory for display, including relevance if available."""
    text = (
        f"{retrieved_memory.memory_text} "
        f"(Categories: {retrieved_memory.categories})"
    )
    if retrieved_memory.score is not None:
        text += f" Relevance: {retrieved_memory.score:.2f}"
    return text


if __name__ == "__main__":
    asyncio.run(create_memory_collection())
