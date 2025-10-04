from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np

app = FastAPI(title="face-health-ai Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection")
    app.state.detector = detector


@app.post("/api/analyze")
async def analyze_face(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await image.read()
    file_bytes = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = app.state.detector
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    for (x, y, w, h) in faces:
        results.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

    return {"faces_count": len(results), "faces": results}


@app.get("/")
async def root():
    return HTMLResponse(
        """<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>FaceIt Prototype</title>
    <script crossorigin src=\"https://unpkg.com/react@18/umd/react.development.js\"></script>
    <script crossorigin src=\"https://unpkg.com/react-dom@18/umd/react-dom.development.js\"></script>
    <link rel=\"stylesheet\" href=\"/static/styles.css\" />
  </head>
  <body>
    <div id=\"root\"></div>
    <script type=\"module\" src=\"/static/app.js\"></script>
  </body>
</html>"""
    )


app.mount("/static", StaticFiles(directory="frontend"), name="static")
