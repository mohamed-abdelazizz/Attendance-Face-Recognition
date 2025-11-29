import argparse
import cv2
from typing import Literal


from database.chroma_manager import ChromaDBManager
from enrollment.enrollment import EnrollmentManager
from face_recognition.detector import FaceDetector
from face_recognition.embedder import FaceEmbedder
from face_recognition.matcher import FaceMatcher
from utils.attendance_logger import AttendanceLogger
from utils.tts import TextToSpeech


# Enrollment
def run_enrollment() -> None:
    """Enroll a new employee by capturing 5 face samples."""
    employee_id = input("Enter employee ID: ").strip()
    employee_name = input("Enter employee name: ").strip()

    if not employee_id or not employee_name:
        print("Employee ID and name are required.")
        return

    # Initialize components
    detector = FaceDetector()
    embedder = FaceEmbedder()
    db_manager = ChromaDBManager()
    enrollment_manager = EnrollmentManager(detector, embedder, db_manager)
    num_samples = enrollment_manager.enroll(employee_id, employee_name)

    if num_samples > 0:
        print(
            f"Enrollment completed for {employee_name} ({employee_id}) with {num_samples} samples.")
    else:
        print("Enrollment cancelled or no samples captured.")


# Helper to draw overlay on webcam frame
def draw_overlay(
    frame,
    bbox,
    name: str,
    mode: Literal["checkin", "checkout"],
    similarity: float | None = None,
):
    """Draw bounding box, labels, and mode text on the frame."""
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = name if similarity is None else f"{name} ({similarity:.2f})"
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show current mode (checkin/checkout)
    cv2.putText(frame, f"Mode: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


# Attendance
def run_attendance() -> None:
    """Run real-time attendance using webcam and face recognition."""
    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = FaceMatcher(threshold=0.5)
    db_manager = ChromaDBManager()
    tts = TextToSpeech()
    logger = AttendanceLogger()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Couldn't open webcam.")

    mode: Literal["checkin", "checkout"] = "checkin"
    last_recognized_id: str | None = None
    last_recognized_mode: str | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            display_name = "Unknown"
            similarity = None
            bbox = None

            # Detect face
            result = detector.detect_and_align(frame)
            if result:
                face_img, bbox = result
                try:
                    emb = embedder.get_embedding(frame)
                except Exception:
                    emb = None

                if emb is not None:
                    embeddings, metadatas = db_manager.get_all_embeddings()
                    match = matcher.find_best_match(emb, embeddings, metadatas)
                    if match:
                        display_name = match.employee_name
                        similarity = match.similarity

                        # Only log & speak if new employee or mode changed
                        if match.employee_id != last_recognized_id or mode != last_recognized_mode:
                            logger.log(match.employee_id,
                                       match.employee_name, mode)
                            if mode == "checkin":
                                tts.speak_async(
                                    f"Welcome, {match.employee_name}")
                            else:
                                tts.speak_async(
                                    f"Goodbye, {match.employee_name}")
                            last_recognized_id = match.employee_id
                            last_recognized_mode = mode
                    else:
                        tts.speak_async("Unknown face detected")
                else:
                    tts.speak_async("Unknown face detected")

            draw_overlay(frame, bbox, display_name, mode, similarity)

            cv2.imshow("Attendance", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # Toggle checkin/checkout mode
                mode = "checkout" if mode == "checkin" else "checkin"

    finally:
        cap.release()
        cv2.destroyAllWindows()


# CLI arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face Recognition Attendance System")
    parser.add_argument(
        "--mode",
        choices=["enroll", "attend"],
        default="attend",
        help="Run in enrollment mode or attendance mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "enroll":
        run_enrollment()
    else:
        run_attendance()
