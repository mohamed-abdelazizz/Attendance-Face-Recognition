# 🚀 Attendance Face Recognition System

This project is a real-time attendance system powered by face recognition technology. It allows for employee enrollment and automated attendance tracking (check-in/check-out) using a webcam or via a REST API. The system utilizes **InsightFace** for state-of-the-art face analysis and **ChromaDB** for efficient vector storage and retrieval.

## ⭐ Features

- **Real-time Face Recognition:** High-accuracy face detection and recognition using InsightFace and ONNX Runtime.
- **Employee Enrollment:** Capture face embeddings and store them in a local vector database.
- **Attendance Tracking:** Automatically log check-in and check-out times.
- **Text-to-Speech (TTS):** Audio feedback greeting employees by name or alerting for unknown faces.
- **REST API:** FastAPI-based interface for remote enrollment and recognition.
- **Vector Database:** Uses ChromaDB to store and manage face embeddings efficiently.
- **CSV Logging:** Attendance records are saved to `attendance_log.csv`.

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

   _Note: Ensure you have the necessary build tools for `insightface` and `onnxruntime` if you encounter installation issues._

## 🖥️ Usage

### 1️⃣ Command Line Interface (CLI)

The system can be run directly from the terminal for local usage with a webcam.

**1. Enrollment Mode:**
Use this mode to register new employees. You will be prompted to enter an ID and Name.

```bash
python main.py --mode enroll
```

**2. Attendance Mode:**
This is the default mode. It opens the webcam to scan for faces.

- Press `c` to toggle between **Check-in** and **Check-out** modes.
- Press `q` to quit.

```bash
python main.py --mode attend
# OR simply
python main.py
```

### 2️⃣ API Usage

You can also run the system as a REST API server.

**1. Start the Server:**

```bash
uvicorn api:app --reload
```

- **API**: `http://127.0.0.1:8000`
- **Interactive Docs (Swagger UI)**: `http://127.0.0.1:8000/docs`
- **Alternative Docs (ReDoc)**: `http://127.0.0.1:8000/redoc`

**2. API Endpoints:**

- **`GET /health`**: Check system status.
- **`POST /enroll`**: Enroll a new employee.
  - Form Data: `employee_id`, `employee_name`, `image` (file)
- **`POST /recognize`**: Recognize a face from an image.
  - Form Data: `image` (file), `mode` (default: "checkin")
- **`GET /employees`**: List all enrolled employees.

## 📁 Project Structure

```
Attendance_Face_Recognition/
├── api.py                  # FastAPI application entry point
├── main.py                 # CLI entry point for Enrollment and Attendance
├── requirements.txt        # Python dependencies
├── attendance_log.csv      # Generated attendance records
├── database/               # ChromaDB management logic
├── enrollment/             # Enrollment process logic
├── face_recognition/       # Core face detection and embedding logic
├── utils/                  # Utilities (Logging, TTS)
└── chroma_db/              # Local storage for ChromaDB
```

## ⚙️ Configuration

- **Thresholds:** Face matching thresholds can be adjusted in `face_recognition/matcher.py` or passed during initialization in `main.py`.
- **Camera:** The default camera index is `0`. Modify `cv2.VideoCapture(0)` in `main.py` if you use an external camera.
