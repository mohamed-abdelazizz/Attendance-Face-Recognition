from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MatchResult:
    employee_id: str
    employee_name: str
    similarity: float


class FaceMatcher:
    """Performs cosine similarity search over embeddings with a configurable threshold."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, str]],
    ) -> Optional[MatchResult]:
        """Return best matching employee or None if below threshold.

        `embeddings` is a list of 1D numpy arrays.
        `metadatas` is a list of dicts containing at least `employee_id` and `employee_name`.
        """
        if not embeddings:
            return None

        # Stack into matrix for cosine similarity computation
        emb_matrix = np.stack(embeddings, axis=0)
        query = query_embedding.reshape(1, -1)
        sims = cosine_similarity(query, emb_matrix)[0]

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < self.threshold:
            return None

        meta = metadatas[best_idx]
        return MatchResult(
            employee_id=meta.get("employee_id", ""),
            employee_name=meta.get("employee_name", ""),
            similarity=best_sim,
        )
