import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np

logger = logging.getLogger("uvicorn")

app = FastAPI(title="face-health-ai Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- helpers ---

def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def skin_mask_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Простейшая маска кожи в HSV; под пороги можно подстроиться под ваши примеры."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # красно-желтая область (учтём wrap по H)
    mask1 = (H <= 0.14)  # ~0..50° в привычной шкале
    mask2 = (H >= 0.97)  # wrap под красный
    mask = (mask1 | mask2) & (S > 0.12) & (S < 0.85) & (V > 0.35)
    return mask.astype(np.uint8)

def flag_by_score(s: float) -> str:
    if s >= 0.6: return "high"
    if s >= 0.3: return "mild"
    return "low"

@app.on_event("startup")
def load_detectors():
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(face_path)
    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection")
    app.state.detector = detector

    eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    eye_detector = cv2.CascadeClassifier(eye_path)
    if eye_detector.empty():
        raise RuntimeError("Failed to load Haar cascade for eye detection")
    app.state.eye_detector = eye_detector


@app.post("/api/analyze")
async def analyze_face(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await image.read()
    file_bytes = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- детекция лиц ---
    detector = app.state.detector
    faces_np = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    faces = [{"x": int(x), "y": int(y), "width": int(fw), "height": int(fh)}
             for (x, y, fw, fh) in faces_np]
    faces_count = len(faces)

    # --- оценка качества кадра ---
    blur = variance_of_laplacian(gray)
    brightness = float(gray.mean())
    face_area_ratio = 0.0
    main_roi = None
    if faces_count > 0:
        # крупнейшее лицо
        x, y, fw, fh = max(faces_np, key=lambda r: r[2]*r[3])
        face_area_ratio = (fw * fh) / (w * h)
        pad = int(0.08 * max(fw, fh))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(w, x + fw + pad); y1 = min(h, y + fh + pad)
        main_roi = frame[y0:y1, x0:x1].copy()

    quality_ok = (blur >= 120.0) and (60.0 <= brightness <= 200.0) and (
        (faces_count == 0) or (face_area_ratio >= 0.12)
    )

    quality = {
        "blur_variance": round(blur, 2),
        "brightness_mean": round(brightness, 2),
        "face_area_ratio": round(face_area_ratio, 3),
        "ok": bool(quality_ok),
        "notes": []
    }
    if blur < 120.0: quality["notes"].append("low_blur")
    if brightness < 60.0: quality["notes"].append("too_dark")
    if brightness > 200.0: quality["notes"].append("too_bright")
    if faces_count > 0 and face_area_ratio < 0.12: quality["notes"].append("face_too_small")

    # --- индикаторы (эвристики) ---
    indicators = {}
    advice = []
    if faces_count == 0:
        advice.append("Лицо не найдено: подойдите ближе к камере и обеспечьте ровное освещение.")
    elif not quality_ok or main_roi is None:
        advice.append("Качество снимка низкое: переснимите при дневном свете, без фильтров/очков.")
    else:
        # маска кожи
        skin = skin_mask_bgr(main_roi).astype(bool)
        roi_rgb = cv2.cvtColor(main_roi, cv2.COLOR_BGR2RGB).astype(np.float32)
        R, G, B = roi_rgb[..., 0], roi_rgb[..., 1], roi_rgb[..., 2]

        # если кожи мало, лучше не делать выводы
        if skin.sum() < 500:
            advice.append("Недостаточно видимой кожи (маски) на лице, попробуйте ещё раз.")
        else:
            R_s = R[skin]; G_s = G[skin]; B_s = B[skin]

            # Покраснение (redness): R vs (G+B)/2
            RI = np.maximum(0.0, R_s - (G_s + B_s) / 2.0) / 255.0
            redness = float(np.clip(RI.mean() if RI.size else 0.0, 0.0, 1.0))

            # Бледность (pallor): низкая цветность (std RGB)
            chroma = np.std(np.stack([R_s, G_s, B_s], axis=-1), axis=-1) / 255.0
            pallor = float(np.clip(1.0 - (chroma.mean() if chroma.size else 0.0), 0.0, 1.0))

            # Желтушность (yellowness) по b* в Lab
            lab = cv2.cvtColor(main_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
            bch = lab[..., 2][skin]
            b_centered = (bch - 128.0) / 127.0  # ~[-1..1]
            yellowness = float(np.clip((b_centered.mean() + 1.0) / 2.0, 0.0, 1.0))

            # Синюшность (cyanosis): B vs R
            CI = np.maximum(0.0, B_s - R_s) / 255.0
            cyanosis = float(np.clip(CI.mean() if CI.size else 0.0, 0.0, 1.0))

            # Склеры: каскад глаз + HSV маска «белков», затем b* в Lab
            eye_detector = app.state.eye_detector
            eye_rects = eye_detector.detectMultiScale(
                cv2.cvtColor(main_roi, cv2.COLOR_BGR2GRAY),
                scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
            )
            sclera_yellow = 0.0
            if len(eye_rects) > 0:
                hsv = cv2.cvtColor(main_roi, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
                S, V = hsv[..., 1], hsv[..., 2]
                sclera_mask = np.zeros(S.shape, dtype=bool)
                for (ex, ey, ew, eh) in eye_rects:
                    ex0, ey0 = max(0, ex), max(0, ey)
                    ex1, ey1 = min(main_roi.shape[1], ex + ew), min(main_roi.shape[0], ey + eh)
                    region = (slice(ey0, ey1), slice(ex0, ex1))
                    region_mask = (S[region] < 0.25) & (V[region] > 0.7)
                    sclera_mask[region][region_mask] = True
                if sclera_mask.sum() > 200:
                    lab_roi = cv2.cvtColor(main_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
                    b_scl = lab_roi[..., 2][sclera_mask]
                    b_c = (b_scl - 128.0) / 127.0
                    sclera_yellow = float(np.clip((b_c.mean() + 1.0) / 2.0, 0.0, 1.0))

            indicators = {
                "skin_redness": {"score": round(redness, 3), "flag": flag_by_score(redness)},
                "skin_pallor": {"score": round(pallor, 3), "flag": flag_by_score(pallor)},
                "jaundice_like": {"score": round(yellowness, 3), "flag": flag_by_score(yellowness)},
                "cyanosis_like": {"score": round(cyanosis, 3), "flag": flag_by_score(cyanosis)},
                "eye_sclera_yellowness": {"score": round(sclera_yellow, 3), "flag": flag_by_score(sclera_yellow)},
            }

    res_obj = {
        "faces_count": faces_count,
        "faces": faces,
        "quality": quality,
        "indicators": indicators,
        "advice": advice,
        "disclaimer": "Предварительная оценка по фото. Не является медицинским диагнозом."
    }

    logger.info(f"faces_count={faces_count} quality_ok={quality['ok']} notes={quality['notes']}")
    return res_obj


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
