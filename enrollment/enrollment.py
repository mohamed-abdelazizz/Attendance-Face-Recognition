import cv2
import numpy as np
from typing import List

from database.chroma_manager import ChromaDBManager
from face_recognition.detector import FaceDetector
from face_recognition.embedder import FaceEmbedder

NUM_IMAGES = 5


class EnrollmentManager:
    """Handles employee enrollment: capture exactly 5 face images."""

    def __init__(self, detector: FaceDetector, embedder: FaceEmbedder, db_manager: ChromaDBManager) -> None:
        self.detector = detector
        self.embedder = embedder
        self.db_manager = db_manager

    def enroll(self, employee_id: str, employee_name: str) -> int:
        """Capture exactly NUM_IMAGES from webcam."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Couldn't open webcam")

        collected_embeddings: List[np.ndarray] = []

        try:
            last_face_img: np.ndarray | None = None

            while len(collected_embeddings) < NUM_IMAGES:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Detect face
                result = self.detector.detect_and_align(frame)
                if result:
                    face_img, bbox = result
                    last_face_img = face_img
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    last_face_img = None

                # Show progress
                cv2.putText(frame, f"Captured {len(collected_embeddings)}/{NUM_IMAGES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, "Align face and press SPACE",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("Enrollment", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == 32:  # SPACE key
                    if last_face_img is None:
                        print("[Enrollment] No face detected")
                        continue

                    try:
                        emb = self.embedder.get_embedding(frame)
                        collected_embeddings.append(emb)
                        print(
                            f"[Enrollment] Captured {len(collected_embeddings)}/{NUM_IMAGES}")
                    except Exception as e:
                        print(f"[Enrollment] Failed embedding: {e}")

        finally:
            cap.release()
            cv2.destroyWindow("Enrollment")

        # Save embeddings
        if collected_embeddings:
            self.db_manager.add_embeddings(
                employee_id, employee_name, collected_embeddings)

        return len(collected_embeddings)
