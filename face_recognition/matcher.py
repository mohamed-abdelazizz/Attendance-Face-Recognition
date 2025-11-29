from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MatchResult:
    employee_id: str
    employee_name: str
    similarity: float


class FaceMatcher:
    """Matches a face embedding against stored embeddings using cosine similarity."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, str]],
    ) -> Optional[MatchResult]:
        """
        Compare query embedding with a list of stored embeddings.
        Return the best match OR None if no match passes the threshold.
        """
        if not embeddings:
            return None

        # Convert list of embeddings into matrix
        # nums_faces × emb_dim (100 , 512)
        emb_matrix = np.stack(embeddings, axis=0)

        # Compute cosine similarity
        cos_sim = cosine_similarity(
            query_embedding.reshape(1, -1), emb_matrix)[0]

        # Find best match
        best_idx = int(np.argmax(cos_sim))
        best_sim = float(cos_sim[best_idx])

        if best_sim < self.threshold:
            return None

        metadata = metadatas[best_idx]

        return MatchResult(
            employee_id=metadata.get("employee_id", ""),
            employee_name=metadata.get("employee_name", ""),
            similarity=best_sim,
        )
