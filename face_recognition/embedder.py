from typing import Any

import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbedder:
    """Computes ArcFace embeddings for aligned face images using InsightFace."""

    def __init__(self, ctx_id: int = 0, det_size=(640, 640)) -> None:
        # Use the same FaceAnalysis backend but focus on embeddings
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Returns a 512-dim normalized embedding for the most prominent face in the image.
        Assumes the image is already roughly aligned/cropped to a single face.
        Returns a 1D numpy array.
        """
        faces = self.app.get(image)
        if not faces:
            raise ValueError("No face detected for embedding.")
        # Choose the largest face
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        face = faces[0]
        # normed_embedding is already L2-normalized
        emb = face.normed_embedding.astype("float32")
        return emb
