# 🚀 FastAPI Face Verification

This project is a complete FastAPI service that verifies whether two uploaded images contain the **face of the same person**.

It uses:
- **DeepFace**
- **ArcFace (high-accuracy embeddings)**
- **RetinaFace (face detection)**



---

## 📁 Project Structure

face-verification-api/
├── main.py
├── requirements.txt
└── README.md

## ⚙️ Setup & Installation

### 1️⃣ Create Project Directory

```bash
mkdir face-verification-api
cd face-verification-api
2️⃣ Create the Required Files
Add:

main.py

requirements.txt

README.md

3️⃣ Create Virtual Environment
Windows
bash
Copy code
python -m venv venv
.\venv\Scripts\activate
macOS / Linux
bash
Copy code
python3 -m venv venv
source venv/bin/activate
4️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
DeepFace will automatically download the ArcFace + RetinaFace models on first run.

▶️ Run the API
bash
Copy code
uvicorn main:app --reload
API runs at:

cpp
Copy code
http://127.0.0.1:8000
📘 Interactive API Docs
Visit:

arduino
Copy code
http://127.0.0.1:8000/docs
Upload two face images → click Execute.

🛰 Example cURL Request
bash
Copy code
curl -X 'POST' \
  'http://127.0.0.1:8000/verify' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file1=@/path/to/image1.jpg' \
  -F 'file2=@/path/to/image2.jpg'
✅ Example Success Response
json
Copy code
{
  "verification_result": "same person",
  "similarity_score": 0.8521,
  "bounding_boxes": {
    "face1_bbox": { "x": 120, "y": 150, "w": 200, "h": 250 },
    "face2_bbox": { "x": 135, "y": 160, "w": 190, "h": 240 }
  },
  "message": "Verification successful"
}
❌ Example Error Response
json
Copy code
{
  "detail": "Error during face processing: Face could not be detected in one or both images."
}
🧰 Features
High-accuracy face verification using ArcFace

Fast and reliable detection using RetinaFace

Simple upload-and-verify API

Clean JSON responses

Swagger UI support

Easy to deploy on cloud or Docker
