from typing import Dict, List, Tuple
import chromadb
import numpy as np
from chromadb.config import Settings


class ChromaDBManager:
    """storing and retrieving face embeddings using ChromaDB."""

    def __init__(
        self,
        db_path: str = "chroma_db",
        collection_name: str = "faces",
    ) -> None:
        # Create a persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        # Get or create a collection for storing embeddings
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )

    def add_embeddings(
        self,
        employee_id: str,
        employee_name: str,
        embeddings: List[np.ndarray],
    ) -> None:
        """Add embeddings for a single employee."""
        if not embeddings:
            return

        ids = []
        vectors = []
        metadatas = []

        for idx, emb in enumerate(embeddings):
            # create a unique id for each embedding
            uid = f"{employee_id}_{idx}"
            ids.append(uid)
            # convert numpy array to list for ChromaDB
            vectors.append(emb.astype(float).tolist())
            # store employee info as metadata
            metadatas.append({
                "employee_id": employee_id,
                "employee_name": employee_name
            })

        # Add all embeddings to the collection
        self.collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)

    def get_all_embeddings(self) -> Tuple[List[np.ndarray], List[Dict[str, str]]]:
        """Return all stored embeddings and their metadata."""
        results = self.collection.get(include=["embeddings", "metadatas"])

        vectors = results.get("embeddings", []) or []
        metadatas = results.get("metadatas", []) or []

        # Convert lists back to numpy arrays to use it later when calculating cosine similarity
        np_vectors = [np.array(v, dtype="float32") for v in vectors]

        print(f"[ChromaDB] Loaded {len(np_vectors)} embeddings")
        return np_vectors, metadatas
