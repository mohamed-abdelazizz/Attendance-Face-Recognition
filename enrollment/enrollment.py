from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from database.chroma_manager import ChromaDBManager
from face_recognition.detector import FaceDetector
from face_recognition.embedder import FaceEmbedder


@dataclass
class EnrollmentConfig:
    num_images: int = 5


class EnrollmentManager:
    """Handles employee enrollment: capture, detect, embed, and store."""

    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        db_manager: ChromaDBManager,
        config: EnrollmentConfig | None = None,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.db_manager = db_manager
        self.config = config or EnrollmentConfig()

    def enroll(self, employee_id: str, employee_name: str) -> int:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam for enrollment.")

        collected_embeddings: List[np.ndarray] = []

        try:
            last_face_img: np.ndarray | None = None

            while len(collected_embeddings) < self.config.num_images:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Detect face every frame for visualization and capture readiness
                result = self.detector.detect_and_align(frame)
                if result is not None:
                    face_img, bbox = result
                    last_face_img = face_img
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    last_face_img = None

                # Always show current progress on screen
                cv2.putText(
                    frame,
                    f"Captured {len(collected_embeddings)}/{self.config.num_images}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                # Hint text
                cv2.putText(
                    frame,
                    "Align your face in the box and press SPACE",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow("Enrollment", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                # SPACE pressed: capture using the current frame (more robust for embedding)
                if key == 32:  # SPACE key
                    print("[Enrollment] SPACE pressed")
                    if last_face_img is None:
                        # No face available to capture at this moment
                        print("[Enrollment] No face detected when SPACE was pressed")
                        continue

                    try:
                        # Use full frame so InsightFace can re-detect and embed robustly
                        emb = self.embedder.get_embedding(frame)
                    except Exception as e:
                        print(f"[Enrollment] Failed to compute embedding for captured face: {e}")
                        continue

                    collected_embeddings.append(emb)
                    print(f"[Enrollment] Captured sample {len(collected_embeddings)}/{self.config.num_images}")
        finally:
            cap.release()
            cv2.destroyWindow("Enrollment")

        if collected_embeddings:
            self.db_manager.add_embeddings(employee_id, employee_name, collected_embeddings)

        # Return how many samples were actually stored
        return len(collected_embeddings)
