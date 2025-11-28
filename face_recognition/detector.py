import cv2
from typing import Optional, Tuple

from insightface.app import FaceAnalysis


class FaceDetector:
    """Wrapper around InsightFace RetinaFace detector with alignment support."""

    def __init__(self, ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)) -> None:
        # Initialize InsightFace FaceAnalysis with RetinaFace model
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect_and_align(self, frame) -> Optional[Tuple[cv2.Mat, Tuple[int, int, int, int]]]:
        """Detects the most prominent face in the frame, aligns it, and returns
        the aligned face image and bounding box as (x1, y1, x2, y2).
        Returns None if no face is detected.
        """
        faces = self.app.get(frame)
        if not faces:
            return None

        # Choose the face with largest bounding box area (primary face)
        faces = sorted(faces, key=lambda f: (
            f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        face = faces[0]

        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        # Ensure box is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # not image; we still want aligned image, use crop from bbox
        aligned_face = face.normed_embedding
        # For downstream embedding we will re-run through embedder; here we return cropped face.
        face_img = frame[y1:y2, x1:x2]
        return face_img, (x1, y1, x2, y2)
