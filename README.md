# 🚀 Attendance Face Recognition System

This project is a real-time attendance system powered by face recognition technology. It allows for employee enrollment and automated attendance tracking (check-in/check-out) using a webcam. The system utilizes **InsightFace** for face analysis and **ChromaDB** for efficient vector storage and retrieval.

## ⭐ Features

- **Real-time Face Recognition:** High-accuracy face detection and recognition using InsightFace and ONNX Runtime.
- **Employee Enrollment:** Capture face embeddings and store them in a chroma vector database.
- **Attendance Tracking:** Automatically log check-in and check-out times.
- **Vector Database:** Uses ChromaDB to store and manage face embeddings efficiently.
- **Text-to-Speech (TTS):** Audio feedback greeting employees by name or alerting for unknown faces.
- **CSV Logging:** Attendance records are saved to `attendance_log.csv`.
- **REST API:** FastAPI-based interface for remote enrollment and recognition.

## 🧰 Tech Stack

| Library           | Usage in Project                                                                          |
| ----------------- | ----------------------------------------------------------------------------------------- |
| **opencv-python** | Captures webcam frames and preprocesses images for face detection.                        |
| **insightface**   | Performs face detection, alignment, and generates high-quality face embeddings (ArcFace). |
| **onnxruntime**   | Runs the InsightFace ONNX models efficiently for fast inference.                          |
| **numpy**         | Handles numerical operations on face embeddings and arrays.                               |
| **pandas**        | Manages attendance logs and employee metadata.                                            |
| **chromadb**      | Stores face embeddings as vectors and provides fast similarity search.                    |
| **pyttsx3**       | Converts system messages to speech (check-in/check-out).                                  |
| **scikit-learn**  | Computes cosine similarity for face matching.                                             |
| **fastapi**       | Backend framework powering the face recognition and attendance API.                       |
| **uvicorn**       | ASGI server used to run the FastAPI backend.                                              |

### 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mohamed-abdelazizz/Attendance_Face_Recognition.git
   cd Attendance_Face_Recognition
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv env
   # On Windows:
   .\env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

### 1️⃣ Command Line Interface (CLI)

The system can be run directly from the terminal for local usage with a webcam.

**1. Enrollment Mode:**
Use this mode to register new employees. You will be prompted to enter an Name and ID.

```bash
python main.py --mode enroll
```

```bash
python main.py --mode attend
# OR simply
python main.py
```

### 2️⃣ API Usage

You can run the system as a REST API server. The API is built with FastAPI and includes interactive docs (Swagger UI).

1. Start the server

```bash
uvicorn api:app --reload
```

- API base: `http://127.0.0.1:8000`
- Interactive docs (Swagger): `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

2. Endpoints — quick summary

| Endpoint     | Method | Description                                                  |
| ------------ | -----: | ------------------------------------------------------------ |
| `/health`    |    GET | Health check (returns status OK).                            |
| `/enroll`    |   POST | Enroll a new employee by uploading 5 face images.            |
| `/recognize` |   POST | Recognize a face from a single image (check-in / check-out). |
| `/employees` |    GET | Return a list of enrolled employees.                         |

Detailed usage below with examples, expected inputs, and response samples.

## GET /health

Returns service health.

Response example (200):

```json
{ "status": "ok" }
```

## POST /enroll — enroll a new employee (5 images)

Register a new employee by uploading exactly five face images. The endpoint accepts a multipart/form-data POST with the following required fields.

Request (multipart/form-data)

- employee_id (string, required): unique ID for the employee
- employee_name (string, required): full name
- image1, image2, image3, image4, image5 (files, required): five separate image files (one face per image is expected)

Behavior & validation

- The endpoint enforces exactly five separate file fields (image1..image5). Omitting a file or sending fewer than five required fields will trigger a validation error.
- Each image must contain a face. If an image cannot be parsed / or an embedding cannot be computed (e.g., no face detected) the endpoint will return HTTP 400 describing which image failed.

Success response

- HTTP 200 —> JSON payload indicating how many embeddings were stored.

Example success response:

```json
{ "stored_samples": 5 }
```

Example: curl (recommended)

```bash
curl -X POST "http://127.0.0.1:8000/enroll" \
   -F "employee_id=20210325" \
   -F "employee_name=Mohamed Abd El-aziz" \
   -F "image1=@/path/to/image1.jpg" \
   -F "image2=@/path/to/image2.jpg" \
   -F "image3=@/path/to/image3.jpg" \
   -F "image4=@/path/to/image4.jpg" \
   -F "image5=@/path/to/image5.jpg"
```

Common error responses

- 400 Bad Request — missing/invalid image or failed embedding for a particular file. Example detail: "Failed to compute embedding for image 3: No face detected"
- 422 Unprocessable Entity — request validation failed (for example, missing required form fields or wrong content type)

## POST /recognize

Recognize a single face image and optionally log the attendance action (checkin/checkout).

Request form fields (multipart/form-data):

- `image` (file, required) — single image containing the face to recognize
- `mode` (string, optional, default "checkin") — either "checkin" or "checkout"

Success response example (recognized):

```json
{
  "employee_id": "20210325",
  "employee_name": "Mohamed Abd El-aziz",
  "similarity": 0.94,
  "mode": "checkin",
  "recognized": true
}
```

If no match is found (face unknown):

```json
{
  "employee_id": null,
  "employee_name": null,
  "similarity": null,
  "mode": "checkin",
  "recognized": false
}
```

Example (curl):

```bash
curl -X POST "http://127.0.0.1:8000/recognize" \
   -F "image=@/path/to/photo.jpg" \
   -F "mode=checkin"
```

## GET /employees

Return a compact list of enrolled employees (unique employee_id + employee_name values).

Response example (200):

```json
[
  { "employee_id": "20210325", "employee_name": "Mohamed Abd El-aziz" },
  { "employee_id": "20210000", "employee_name": "Full Name" }
]
```

## Tips

- Use the interactive Swagger UI (`/docs`) to try endpoints and quickly upload files for testing.
- If you get a 400 response for an image, try a different photo where the face is clearly visible.

## 📁 Project structure (complete tree)

```
Attendance_Face_Recognition/
├── api.py                      # FastAPI application entry point
├── main.py                     # CLI entry point for Enrollment and Attendance
├── requirements.txt            # Python dependencies
├── attendance_log.csv          # Generated attendance records (CSV written at runtime)
├── README.md
├── .gitignore                  # Ignored files for git
|
├── database/                   # ChromaDB management logic
│   └── chroma_manager.py       # Add / fetch embeddings and metadata
|
├── enrollment/                 # Enrollment process logic
│   └── enrollment.py           # EnrollmentManager: capture frames, compute embeddings, save to chroma DB
|
├── face_recognition/           # Core face detection and embedding logic
│   ├── detector.py
│   ├── embedder.py
│   └── matcher.py
|
├── utils/                      # Utilities (CSV Logging, TTS)
│   ├── attendance_logger.py
│   └── tts.py                  #
|
└── chroma_db/                  # Local vector store
   ├── chroma.sqlite3
```

## ⚙️ Configuration

- **Thresholds:** Face matching thresholds can be adjusted in `face_recognition/matcher.py` or passed during initialization in `main.py`.
- **Camera:** The default camera index is `0`. Modify `cv2.VideoCapture(0)` in `main.py` if you use an external camera.
