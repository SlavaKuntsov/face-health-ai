import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np

# Pillow опционален (для EXIF-ориентации)
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

logger = logging.getLogger("uvicorn")

app = FastAPI(title="face-health-ai Prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- helpers (ваши + новые) ---

def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def skin_mask_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Простейшая маска кожи в HSV; под пороги можно подстроиться под ваши примеры."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # красно-желтая область (учтём wrap по H)
    mask1 = (H <= 0.14)  # ~0..50°
    mask2 = (H >= 0.97)
    mask = (mask1 | mask2) & (S > 0.12) & (S < 0.85) & (V > 0.35)
    return mask.astype(np.uint8)

def flag_by_score(s: float) -> str:
    if s >= 0.6: return "high"
    if s >= 0.3: return "mild"
    return "low"

def fix_orientation_and_decode(image_bytes: bytes) -> np.ndarray | None:
    """Корректно читаем JPEG/PNG с учётом EXIF-ориентации."""
    if PIL_AVAILABLE:
        try:
            img = Image.open(BytesIO(image_bytes))   # <-- было: Image.open(np.frombuffer(...))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            arr = np.asarray(img)  # RGB
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"EXIF transpose failed, fallback to cv2: {e}")
    # fallback
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def resize_keep_ratio(frame: np.ndarray, max_width: int = 1024) -> Tuple[np.ndarray, float]:
    """Возвращает (scaled_frame, scale). scale = new_w / orig_w."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    new_w = max_width
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

def nms_boxes(boxes: List[Tuple[int,int,int,int]], scores: List[float], iou_thresh: float = 0.4) -> List[int]:
    """Простая NMS. boxes: [x1,y1,x2,y2]"""
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes_np[:,0], boxes_np[:,1], boxes_np[:,2], boxes_np[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_np.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def map_to_original_scale(results_proc: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    """Пересчёт координат из уменьшенного кадра к исходному размеру."""
    if scale == 1.0:
        return results_proc
    out = []
    inv = 1.0 / scale
    for r in results_proc:
        x = int(round(r["x"] * inv))
        y = int(round(r["y"] * inv))
        w = int(round(r["width"] * inv))
        h = int(round(r["height"] * inv))
        rr = {"x": x, "y": y, "width": w, "height": h}
        if "confidence" in r: rr["confidence"] = float(r["confidence"])
        out.append(rr)
    return out

def _clip_box(x1, y1, x2, y2, W, H):
    return max(0, x1), max(0, y1), min(W-1, x2), min(H-1, y2)

def _skin_ratio(frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    if x2 <= x1 or y2 <= y1: 
        return 0.0
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0: 
        return 0.0
    mask = skin_mask_bgr(roi).astype(bool)
    return float(mask.mean())

def _has_eye(frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    # очень дёшево: CLAHE по серому + маленький каскад глаз
    gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if gray.size == 0 or gray.shape[0] < 24 or gray.shape[1] < 24:
        return False
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    eye_det = app.state.eye_detector
    eyes = eye_det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(12, 12))
    return len(eyes) >= 1

def post_filter_faces(frame_bgr: np.ndarray, faces: list, brightness: float) -> list:
    """
    faces: [{x,y,width,height, [confidence]}] в координатах ИСХОДНОГО изображения.
    Возвращает отфильтрованный список.
    """
    H, W = frame_bgr.shape[:2]
    # Тюнимые параметры (стартовые значения адекватны для большинства камер):
    MIN_AREA_FRAC = 0.005   # мин. площадь бокса относительно кадра (0.5%)
    MIN_SKIN_FRAC = 0.15    # мин. доля «кожи» в боксе
    ASPECT_MIN, ASPECT_MAX = 0.6, 1.6  # допустимое соотношение сторон h/w (лицо не слишком вытянутое)

    # при тёмном кадре допустим меньше skin (шум сильнее)
    if brightness < 70:
        MIN_SKIN_FRAC = 0.10

    filtered = []
    for f in faces:
        x, y, w, h = int(f["x"]), int(f["y"]), int(f["width"]), int(f["height"])
        if w <= 0 or h <= 0: 
            continue

        area = w * h
        if area / float(W * H) < MIN_AREA_FRAC:
            continue

        aspect = h / float(w)
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue

        x1, y1, x2, y2 = _clip_box(x, y, x + w, y + h, W, H)
        skin_frac = _skin_ratio(frame_bgr, x1, y1, x2, y2)
        if skin_frac < MIN_SKIN_FRAC:
            # шанс спасения: если глаз найден — оставим (полезно при желтоватом/неоднородном свете)
            if not _has_eye(frame_bgr, x1, y1, x2, y2):
                continue

        filtered.append(f)

    return filtered

# --- загрузка моделей ---

@app.on_event("startup")
def load_detectors():
    # Haar (лицо)
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    haar_face = cv2.CascadeClassifier(face_path)
    if haar_face.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection")
    app.state.haar_face = haar_face

    # Haar (глаза)
    eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    eye_detector = cv2.CascadeClassifier(eye_path)
    if eye_detector.empty():
        raise RuntimeError("Failed to load Haar cascade for eye detection")
    app.state.eye_detector = eye_detector

    # >>> КЛЮЧЕВОЕ: путь к models рядом с ЭТИМ файлом (main.py)
    MODELS_DIR = Path(__file__).resolve().parent / "models"
    proto = MODELS_DIR / "deploy.prototxt"
    weights = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

    logger.info(f"DNN paths: proto={proto} exists={proto.exists()}, weights={weights} exists={weights.exists()}")

    if not proto.exists() or not weights.exists():
        logger.warning("DNN face model files are missing; only Haar will be used")
        app.state.dnn = None
        return

    try:
        net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        # (опционально) CUDA:
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        app.state.dnn = net
        logger.info("DNN face detector loaded")
    except Exception as e:
        logger.exception(f"Failed to load DNN model: {e}")
        app.state.dnn = None

# --- детекторы ---

def detect_faces_dnn(frame_bgr: np.ndarray, conf_thresh: float = 0.6):
    net = getattr(app.state, "dnn", None)
    if net is None:
        return []

    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)),
        scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False, crop=False
    )
    try:
        net.setInput(blob)
        det = net.forward()  # [1,1,N,7]
    except Exception as e:
        logger.exception(f"DNN forward failed, fallback to Haar: {e}")
        return []  # пустой список -> в хендлере сработает fallback

    raw_boxes, raw_scores = [], []
    for i in range(det.shape[2]):
        conf = float(det[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([w, h, w, h], dtype=np.float32)).astype(int)
        x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
        if x2 > x1 and y2 > y1:
            raw_boxes.append((x1, y1, x2, y2))
            raw_scores.append(conf)

    keep = nms_boxes(raw_boxes, raw_scores, iou_thresh=0.4)
    return [{
        "x": int(raw_boxes[i][0]),
        "y": int(raw_boxes[i][1]),
        "width": int(raw_boxes[i][2] - raw_boxes[i][0]),
        "height": int(raw_boxes[i][3] - raw_boxes[i][1]),
        "confidence": float(raw_scores[i])
    } for i in keep]


def detect_faces_haar(gray_proc: np.ndarray) -> List[Dict[str, Any]]:
    """Haar как резерв (на контраст-улучшенном изображении)."""
    haar = app.state.haar_face
    faces = haar.detectMultiScale(
        gray_proc,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for (x, y, w, h) in faces]

# --- API ---

@app.post("/api/analyze")
async def analyze_face(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await image.read()
    frame = fix_orientation_and_decode(data)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    H, W = frame.shape[:2]

    # масштабируем для стабильности/скорости
    proc, scale = resize_keep_ratio(frame, max_width=1024)

    # 1) DNN
    dnn_results_proc = detect_faces_dnn(proc, conf_thresh=0.7)
    detector_used = "dnn" if dnn_results_proc else "haar"

    # 2) Если нет – Haar на CLAHE
    if not dnn_results_proc:
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        haar_results_proc = detect_faces_haar(gray)
        results_proc = haar_results_proc
    else:
        results_proc = dnn_results_proc

    # Пересчитываем координаты в исходный размер
    # Пересчитываем координаты в исходный размер
    faces = map_to_original_scale(results_proc, scale)

    # ВАЖНО: пост-фильтр
    faces = post_filter_faces(frame, faces, brightness=float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))

    faces_count = len(faces)

    # --- оценка качества кадра ---
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = variance_of_laplacian(gray_full)
    brightness = float(gray_full.mean())
    face_area_ratio = 0.0
    main_roi = None

    if faces_count > 0:
        # крупнейшее лицо на исходном размере
        bx = max(faces, key=lambda r: r["width"] * r["height"])
        x, y, fw, fh = bx["x"], bx["y"], bx["width"], bx["height"]
        face_area_ratio = (fw * fh) / (W * H)
        pad = int(0.08 * max(fw, fh))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + fw + pad); y1 = min(H, y + fh + pad)
        main_roi = frame[y0:y1, x0:x1].copy()

    quality_ok = (blur >= 120.0) and (60.0 <= brightness <= 200.0) and (
        (faces_count == 0) or (face_area_ratio >= 0.12)
    )

    quality = {
        "blur_variance": round(blur, 2),
        "brightness_mean": round(brightness, 2),
        "face_area_ratio": round(face_area_ratio, 3),
        "ok": bool(quality_ok),
        "notes": [],
        "detector": detector_used
    }
    if blur < 120.0: quality["notes"].append("low_blur")
    if brightness < 60.0: quality["notes"].append("too_dark")
    if brightness > 200.0: quality["notes"].append("too_bright")
    if faces_count > 0 and face_area_ratio < 0.12: quality["notes"].append("face_too_small")

    # --- индикаторы (эвристики) ---
    indicators: Dict[str, Any] = {}
    advice: List[str] = []

    if faces_count == 0:
        advice.append("Лицо не найдено: подойдите ближе к камере и обеспечьте ровное освещение.")
    elif not quality_ok or main_roi is None:
        advice.append("Качество снимка низкое: переснимите при дневном свете, без фильтров/очков.")
    else:
        skin = skin_mask_bgr(main_roi).astype(bool)
        roi_rgb = cv2.cvtColor(main_roi, cv2.COLOR_BGR2RGB).astype(np.float32)
        R, G, B = roi_rgb[..., 0], roi_rgb[..., 1], roi_rgb[..., 2]

        if skin.sum() < 500:
            advice.append("Недостаточно видимой кожи (маски) на лице, попробуйте ещё раз.")
        else:
            R_s = R[skin]; G_s = G[skin]; B_s = B[skin]

            # Покраснение
            RI = np.maximum(0.0, R_s - (G_s + B_s) / 2.0) / 255.0
            redness = float(np.clip(RI.mean() if RI.size else 0.0, 0.0, 1.0))

            # Бледность
            chroma = np.std(np.stack([R_s, G_s, B_s], axis=-1), axis=-1) / 255.0
            pallor = float(np.clip(1.0 - (chroma.mean() if chroma.size else 0.0), 0.0, 1.0))

            # Желтушность по b* (Lab)
            lab = cv2.cvtColor(main_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
            bch = lab[..., 2][skin]
            b_centered = (bch - 128.0) / 127.0
            yellowness = float(np.clip((b_centered.mean() + 1.0) / 2.0, 0.0, 1.0))

            # Синюшность
            CI = np.maximum(0.0, B_s - R_s) / 255.0
            cyanosis = float(np.clip(CI.mean() if CI.size else 0.0, 0.0, 1.0))

            # Склеры (через каскад глаз)
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
        "faces": faces,              # координаты в ИСХОДНОМ размере изображения
        "quality": quality,
        "indicators": indicators,
        "advice": advice,
        "disclaimer": "Предварительная оценка по фото. Не является медицинским диагнозом."
    }

    logger.info(f"faces_count={faces_count} detector={quality['detector']} quality_ok={quality['ok']} notes={quality['notes']}")
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
