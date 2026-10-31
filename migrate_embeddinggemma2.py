"""Copy an existing Qdrant collection into a new EmbeddingGemma 2 collection.

The source collection is never modified or deleted. Set EMBEDDING_MODEL only
after this command completes successfully. QDRANT_COLLECTION_EG2 may be set
beforehand to choose the destination name.
"""

import os
import sys
import argparse
from typing import Any

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


MODEL_ID = "google/embeddinggemma-2"
DIMENSION = 256
SCROLL_PAGE_SIZE = 256
EMBED_BATCH_SIZE = 32


def get_single_vector_config(collection_info: Any) -> Any:
    vectors = collection_info.config.params.vectors
    if isinstance(vectors, dict):
        raise RuntimeError("Named-vector source collections are not supported by this migration")
    return vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted migration into a compatible destination collection",
    )
    args = parser.parse_args()
    load_dotenv()
    source_name = os.getenv("QDRANT_COLLECTION", "document_chunks")
    target_name = os.getenv("QDRANT_COLLECTION_EG2", f"{source_name}_embeddinggemma2")
    qdrant_url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise RuntimeError("QDRANT_URL must be set")
    if source_name == target_name:
        raise RuntimeError("Source and destination collection names must differ")

    client = QdrantClient(url=qdrant_url, api_key=api_key.strip() if api_key else None)
    if not client.collection_exists(source_name):
        raise RuntimeError(f"Source collection {source_name!r} does not exist")
    source_info = client.get_collection(source_name)
    source_vectors = get_single_vector_config(source_info)
    source_count = client.count(source_name, exact=True).count
    if source_count == 0:
        raise RuntimeError("Source collection is empty; refusing to create an empty migration")

    # The model is local; no hosted embedding API is used. EmbeddingGemma 2
    # projects to 256 dimensions to reduce vector storage and search work.
    model = SentenceTransformer(
        MODEL_ID,
        config_kwargs={"vision_config": None, "audio_config": None},
        truncate_dim=DIMENSION,
    )

    target_exists = client.collection_exists(target_name)
    if target_exists:
        if not args.resume:
            raise RuntimeError(
                f"Destination {target_name!r} already exists. It was left untouched; "
                "use --resume only if this is a partial migration created by this script."
            )
        target_vectors = get_single_vector_config(client.get_collection(target_name))
        if target_vectors.size != DIMENSION or target_vectors.distance != source_vectors.distance:
            raise RuntimeError("Existing destination vector size or distance is incompatible; it was left untouched")
        print(f"Resuming into existing compatible destination {target_name!r}", flush=True)
    else:
        client.create_collection(
            collection_name=target_name,
            vectors_config=models.VectorParams(size=DIMENSION, distance=source_vectors.distance),
        )

    offset = None
    scanned = 0
    embedded_this_run = 0
    source_ids = set()
    while True:
        points, next_offset = client.scroll(
            collection_name=source_name,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break

        prepared = []
        for point in points:
            source_ids.add(point.id)
            payload = point.payload or {}
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"Source point {point.id!r} has no non-empty text payload")
            title = payload.get("filename") or payload.get("source") or "none"
            prepared.append((point.id, payload, f"title: {title} | text: {text}"))

        if target_exists:
            existing_points = client.retrieve(
                collection_name=target_name,
                ids=[point_id for point_id, _, _ in prepared],
                with_payload=True,
                with_vectors=True,
            )
            existing_by_id = {point.id: point for point in existing_points}
            prepared = [
                item
                for item in prepared
                if (
                    (existing := existing_by_id.get(item[0])) is None
                    or existing.payload != item[1]
                    or np.asarray(existing.vector).shape != (DIMENSION,)
                    or not np.isfinite(np.asarray(existing.vector)).all()
                )
            ]

        if prepared:
            vectors = model.encode(
                [item[2] for item in prepared],
                prompt_name="Document",
                normalize_embeddings=True,
                batch_size=EMBED_BATCH_SIZE,
                show_progress_bar=False,
            )
            if vectors.ndim != 2 or vectors.shape != (len(prepared), DIMENSION):
                raise RuntimeError(f"Unexpected embedding shape {vectors.shape}; expected (*, {DIMENSION})")

            client.upsert(
                collection_name=target_name,
                points=[
                    models.PointStruct(id=point_id, vector=np.asarray(vector).tolist(), payload=payload)
                    for (point_id, payload, _), vector in zip(prepared, vectors)
                ],
                wait=True,
            )
            embedded_this_run += len(prepared)

        scanned += len(points)
        print(
            f"Scanned {scanned}/{source_count} source points; embedded {embedded_this_run} new points this run",
            flush=True,
        )
        if next_offset is None:
            break
        offset = next_offset

    destination_count = client.count(target_name, exact=True).count
    destination_ids = set()
    destination_offset = None
    while True:
        destination_points, next_destination_offset = client.scroll(
            collection_name=target_name,
            limit=SCROLL_PAGE_SIZE,
            offset=destination_offset,
            with_payload=False,
            with_vectors=False,
        )
        destination_ids.update(point.id for point in destination_points)
        if next_destination_offset is None:
            break
        destination_offset = next_destination_offset

    if (
        scanned != source_count
        or len(source_ids) != source_count
        or destination_count != source_count
        or source_ids != destination_ids
    ):
        raise RuntimeError(
            f"Verification failed: source={source_count}, scanned={scanned}, "
            f"destination={destination_count}, missing={len(source_ids - destination_ids)}, "
            f"extra={len(destination_ids - source_ids)}. "
            f"Source collection {source_name!r} remains intact."
        )

    print(
        f"Verified {destination_count} points in {target_name!r}. "
        f"Original collection {source_name!r} was not modified."
    )
    print("To switch the app after validation, set EMBEDDING_MODEL=google/embeddinggemma-2")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Migration stopped: {exc}", file=sys.stderr)
        print("The source collection is retained. Inspect the destination before retrying.", file=sys.stderr)
        raise SystemExit(1)
