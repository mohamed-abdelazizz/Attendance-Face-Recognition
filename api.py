from typing import Optional, List
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from database.chroma_manager import ChromaDBManager
from face_recognition.embedder import FaceEmbedder
from face_recognition.matcher import FaceMatcher
from utils.attendance_logger import AttendanceLogger
from utils.tts import TextToSpeech

app = FastAPI(title="Attendance Face Recognition API")

# Initialize components
embedder = FaceEmbedder()
matcher = FaceMatcher()
chroma_manager = ChromaDBManager()
logger = AttendanceLogger()
tts = TextToSpeech()


class HealthResponse(BaseModel):
    status: str


class EnrollResponse(BaseModel):
    stored_samples: int


class RecognizeResponse(BaseModel):
    employee_id: Optional[str]
    employee_name: Optional[str]
    similarity: Optional[float]
    mode: str
    recognized: bool


class EmployeeInfo(BaseModel):
    employee_id: str
    employee_name: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok")

# convert UploadFile image to numpy image


def _load_image_to_ndarray(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Empty image file.")
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid image data.")
    return img


@app.post("/enroll", response_model=EnrollResponse)
async def enroll_employee(
    employee_id: str = Form(...),
    employee_name: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...),
    image4: UploadFile = File(...),
    image5: UploadFile = File(...)
) -> EnrollResponse:
    images = [image1, image2, image3, image4, image5]
    if len(images) != 5:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Exactly 5 images are required.")

    embeddings = []

    for image in images:
        img = _load_image_to_ndarray(image)
        try:
            emb = embedder.get_embedding(img)
            embeddings.append(emb)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Failed to compute embedding: {e}")

    chroma_manager.add_embeddings(employee_id, employee_name, embeddings)
    return EnrollResponse(stored_samples=len(embeddings))


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize_employee(
    image: UploadFile = File(...),
    mode: str = Form("checkin"),
) -> RecognizeResponse:
    if mode not in {"checkin", "checkout"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="mode must be 'checkin' or 'checkout'")

    img = _load_image_to_ndarray(image)

    try:
        emb = embedder.get_embedding(img)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Failed to compute embedding: {e}")

    embeddings, metadatas = chroma_manager.get_all_embeddings()
    match = matcher.find_best_match(emb, embeddings, metadatas)

    if match is None:
        tts.speak_async("Unknown face detected")
        return RecognizeResponse(employee_id=None,
                                 employee_name=None,
                                 similarity=None,
                                 mode=mode,
                                 recognized=False)

    logger.log(match.employee_id, match.employee_name, mode)
    if mode == "checkin":
        tts.speak_async(f"Welcome, {match.employee_name}")
    else:
        tts.speak_async(f"Goodbye, {match.employee_name}")

    return RecognizeResponse(employee_id=match.employee_id,
                             employee_name=match.employee_name,
                             similarity=match.similarity,
                             mode=mode,
                             recognized=True)


@app.get("/employees", response_model=List[EmployeeInfo])
async def list_employees() -> List[EmployeeInfo]:
    embeddings, metadatas = chroma_manager.get_all_embeddings()
    seen: dict[str, str] = {}
    for meta in metadatas:
        emp_id = meta.get("employee_id", "")
        emp_name = meta.get("employee_name", "")
        if emp_id and emp_id not in seen:
            seen[emp_id] = emp_name

    return [EmployeeInfo(employee_id=eid, employee_name=name) for eid, name in seen.items()]
