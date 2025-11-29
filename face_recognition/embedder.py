import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbedder:
    """Generates 512-d face embeddings using InsightFace ArcFace."""

    def __init__(self, ctx_id: int = 0, det_size=(640, 640)) -> None:
        # Load model for embeddings
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract a 512-d embedding from a cropped face image.
        """
        faces = self.app.get(image)
        if not faces:
            return None

        # Pick the largest detected face
        faces = sorted(faces, key=lambda f: (
            f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        face = faces[0]

        # Return embedding
        return face.normed_embedding.astype("float32")
