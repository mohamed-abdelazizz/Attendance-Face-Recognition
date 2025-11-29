import cv2
from typing import Optional, Tuple
from insightface.app import FaceAnalysis


class FaceDetector:
    """face detector using InsightFace RetinaFace."""

    def __init__(self, ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)) -> None:
        # Load the face detection model
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect_and_align(self, frame) -> Optional[Tuple[cv2.Mat, Tuple[int, int, int, int]]]:
        """
        Detect the largest face in the frame 
        Return:
        - Cropped face image
        - Bounding box (x1, y1, x2, y2)
        Return None if no face is found.
        """
        faces = self.app.get(frame)
        if not faces:
            return None

        # Pick the largest detected face
        faces = sorted(faces, key=lambda f: (
            f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        face = faces[0]

        # Extract bounding box
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Make sure the box is inside the image
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Crop the face from the frame
        face_img = frame[y1:y2, x1:x2]

        return face_img, (x1, y1, x2, y2)
