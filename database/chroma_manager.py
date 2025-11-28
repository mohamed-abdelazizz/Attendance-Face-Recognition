from typing import Dict, List, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings


class ChromaDBManager:
    """Abstraction over ChromaDB for storing and querying face embeddings."""

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "face_embeddings",
    ) -> None:
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_embeddings(
        self,
        employee_id: str,
        employee_name: str,
        embeddings: List[np.ndarray],
    ) -> None:
        """Add multiple embeddings for a single employee.

        Each embedding is stored with unique ID and metadata.
        """
        if not embeddings:
            return

        ids = []
        vectors: List[List[float]] = []
        metadatas: List[Dict[str, str]] = []
        for idx, emb in enumerate(embeddings):
            uid = f"{employee_id}_{idx}"
            ids.append(uid)
            vectors.append(emb.astype(float).tolist())
            metadatas.append(
                {
                    "employee_id": employee_id,
                    "employee_name": employee_name,
                }
            )

        self.collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)

    def get_all_embeddings(self) -> Tuple[List[np.ndarray], List[Dict[str, str]]]:
        """Return all stored embeddings and their metadata."""
        # Explicitly include embeddings and metadatas to avoid version-specific defaults
        results = self.collection.get(include=["embeddings", "metadatas"])
        vectors = results.get("embeddings", []) or []
        metadatas = results.get("metadatas", []) or []
        print(f"[ChromaDB] get_all_embeddings: {len(vectors)} embeddings")
        np_vectors = [np.array(v, dtype="float32") for v in vectors]
        return np_vectors, metadatas
