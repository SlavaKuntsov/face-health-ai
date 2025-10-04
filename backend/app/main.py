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

# ---------- helpers ----------

def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def skin_mask_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Простейшая маска кожи в HSV; под пороги можно подстроиться под ваши примеры."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask1 = (H <= 0.14)           # ~0..50°
    mask2 = (H >= 0.97)           # wrap к красному
    mask = (mask1 | mask2) & (S > 0.12) & (S < 0.85) & (V > 0.35)
    return mask.astype(np.uint8)

def triage_label(score: float) -> str:
    """3 уровня: болен / сомнительно / здоров (по risk_score 0..1)."""
    if score >= 0.60:
        return "болен"
    if score >= 0.35:
        return "сомнительно"
    return "здоров"

def fix_orientation_and_decode(image_bytes: bytes) -> np.ndarray | None:
    """Корректно читаем JPEG/PNG с учётом EXIF-ориентации."""
    if PIL_AVAILABLE:
        try:
            img = Image.open(BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            arr = np.asarray(img)  # RGB
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"EXIF transpose failed, fallback to cv2: {e}")
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
    gray = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if gray.size == 0 or gray.shape[0] < 24 or gray.shape[1] < 24:
        return False
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    eye_det = app.state.eye_detector
    eyes = eye_det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(12, 12))
    return len(eyes) >= 1

def post_filter_faces(frame_bgr: np.ndarray, faces: list, brightness: float) -> list:
    """Возвращает отфильтрованный список: [{x,y,width,height, [confidence]}]"""
    H, W = frame_bgr.shape[:2]
    MIN_AREA_FRAC = 0.005
    MIN_SKIN_FRAC = 0.15
    ASPECT_MIN, ASPECT_MAX = 0.6, 1.6

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
        if skin_frac < MIN_SKIN_FRAC and not _has_eye(frame_bgr, x1, y1, x2, y2):
            continue
        filtered.append(f)
    return filtered

# ---------- загрузка моделей ----------

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

    MODELS_DIR = Path(__file__).resolve().parent / "models"

    # DNN (SSD ResNet10 Caffe) — опционально (не прерываем загрузку при отсутствии)
    proto = MODELS_DIR / "deploy.prototxt"
    weights = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    app.state.dnn = None
    logger.info(f"DNN paths: proto={proto.exists()}, weights={weights.exists()}")
    if proto.exists() and weights.exists():
        try:
            net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
            app.state.dnn = net
            logger.info("DNN face detector loaded")
        except Exception as e:
            logger.exception(f"Failed to load DNN model: {e}")

    # YuNet (ONNX) — основной
    yn_path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    app.state.yunet = None
    try:
        if yn_path.exists():
            # API бывает двух форм: FaceDetectorYN_create или FaceDetectorYN.create
            if hasattr(cv2, "FaceDetectorYN_create"):
                yn = cv2.FaceDetectorYN_create(
                    str(yn_path), "", (320, 320), 0.6, 0.3, 500
                )
            elif hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
                yn = cv2.FaceDetectorYN.create(
                    str(yn_path), "", (320, 320), 0.6, 0.3, 500
                )
            else:
                yn = None
            app.state.yunet = yn
            logger.info("YuNet loaded" if yn is not None else "YuNet API not available in this OpenCV build")
        else:
            logger.warning("YuNet model not found; detector will fallback to DNN/Haar")
    except Exception as e:
        logger.exception(f"YuNet init failed: {e}")
        app.state.yunet = None


# ---------- детекторы ----------

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
        return []

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
    haar = app.state.haar_face
    faces = haar.detectMultiScale(
        gray_proc, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for (x, y, w, h) in faces]


def detect_faces_yunet(frame_bgr: np.ndarray, conf_thresh: float = 0.6) -> List[Dict[str, Any]]:
    yn = getattr(app.state, "yunet", None)
    if yn is None:
        return []
    h, w = frame_bgr.shape[:2]
    # Важно: перед каждым вызовом обновлять входной размер
    try:
        if hasattr(yn, "setInputSize"):
            yn.setInputSize((w, h))
        retval, faces = yn.detect(frame_bgr)
    except Exception as e:
        logger.exception(f"YuNet detect failed: {e}")
        return []

    if faces is None or len(faces) == 0:
        return []

    out = []
    # Формат: [x, y, w, h, 5*landmarks..., score] (N x 15)
    for r in faces:
        x, y, ww, hh = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        score = float(r[-1])
        if score < conf_thresh or ww <= 0 or hh <= 0:
            continue
        out.append({"x": x, "y": y, "width": ww, "height": hh, "confidence": score})
    return out


# ---------- per-face оценка ----------

def compute_indicators_for_roi(roi_bgr: np.ndarray) -> Dict[str, float]:
    skin = skin_mask_bgr(roi_bgr).astype(bool)
    res = {"skin_pix": int(skin.sum()), "sclera_pix": 0}
    if skin.sum() < 300:
        # слишком мало кожи — вернём нули, дальше это учтём по весам
        res.update({
            "skin_redness": 0.0, "skin_pallor": 0.0,
            "jaundice_like": 0.0, "cyanosis_like": 0.0,
            "eye_sclera_yellowness": 0.0
        })
        return res

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = roi_rgb[..., 0], roi_rgb[..., 1], roi_rgb[..., 2]
    R_s, G_s, B_s = R[skin], G[skin], B[skin]

    # Покраснение
    RI = np.maximum(0.0, R_s - (G_s + B_s) / 2.0) / 255.0
    redness = float(np.clip(RI.mean() if RI.size else 0.0, 0.0, 1.0))

    # Бледность (низкая цветность)
    chroma = np.std(np.stack([R_s, G_s, B_s], axis=-1), axis=-1) / 255.0
    pallor = float(np.clip(1.0 - (chroma.mean() if chroma.size else 0.0), 0.0, 1.0))

    # Желтушность по b* (Lab)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bch = lab[..., 2][skin]
    b_centered = (bch - 128.0) / 127.0
    yellowness = float(np.clip((b_centered.mean() + 1.0) / 2.0, 0.0, 1.0))

    # Синюшность
    CI = np.maximum(0.0, B_s - R_s) / 255.0
    cyanosis = float(np.clip(CI.mean() if CI.size else 0.0, 0.0, 1.0))

    # Склеры (через каскад глаз + HSV фильтр)
    eye_detector = app.state.eye_detector
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    eyes = eye_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    sclera_yellow = 0.0
    sclera_pix = 0
    if len(eyes) > 0:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
        S, V = hsv[..., 1], hsv[..., 2]
        sclera_mask = np.zeros(S.shape, dtype=bool)
        for (ex, ey, ew, eh) in eyes:
            ex0, ey0 = max(0, ex), max(0, ey)
            ex1, ey1 = min(roi_bgr.shape[1], ex + ew), min(roi_bgr.shape[0], ey + eh)
            region = (slice(ey0, ey1), slice(ex0, ex1))
            region_mask = (S[region] < 0.25) & (V[region] > 0.7)
            sclera_mask[region][region_mask] = True
        sclera_pix = int(sclera_mask.sum())
        if sclera_pix > 200:
            lab_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            b_scl = lab_roi[..., 2][sclera_mask]
            b_c = (b_scl - 128.0) / 127.0
            sclera_yellow = float(np.clip((b_c.mean() + 1.0) / 2.0, 0.0, 1.0))

    res.update({
        "skin_redness": round(redness, 3),
        "skin_pallor": round(pallor, 3),
        "jaundice_like": round(yellowness, 3),
        "cyanosis_like": round(cyanosis, 3),
        "eye_sclera_yellowness": round(sclera_yellow, 3),
        "sclera_pix": sclera_pix
    })
    return res

def risk_from_indicators(ind: Dict[str, float]) -> float:
    """Взвешенное среднее с авто-нормировкой по доступным признакам."""
    weights = {
        "skin_redness": 0.15,
        "skin_pallor": 0.25,
        "jaundice_like": 0.30,
        "cyanosis_like": 0.20,
        "eye_sclera_yellowness": 0.30,
    }
    s = 0.0
    w = 0.0
    for k, wt in weights.items():
        if k in ind:
            s += ind[k] * wt
            w += wt
    if w <= 0:
        return 0.0
    return float(np.clip(s / w, 0.0, 1.0))

def assess_face(frame_bgr: np.ndarray, box: Dict[str, int], img_wh: Tuple[int,int]) -> Dict[str, Any]:
    """ROI с паддингом, локальные метрики качества, индикаторы, итоговая оценка."""
    H, W = img_wh[1], img_wh[0]
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    pad = int(0.08 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    roi = frame_bgr[y0:y1, x0:x1].copy()

    # локальное качество
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_local = variance_of_laplacian(gray)
    bright_local = float(gray.mean())
    area_frac = (w * h) / float(W * H)

    quality_local = {
        "blur_variance": round(blur_local, 2),
        "brightness_mean": round(bright_local, 2),
        "face_area_ratio": round(area_frac, 4),
        "ok": (blur_local >= 120.0) and (60.0 <= bright_local <= 200.0) and (area_frac >= 0.01)
    }

    ind = compute_indicators_for_roi(roi)
    # не учитываем sclera, если её мало найдено
    if ind.get("sclera_pix", 0) < 200 and "eye_sclera_yellowness" in ind:
        ind["eye_sclera_yellowness"] = 0.0

    risk = risk_from_indicators(ind)
    label = triage_label(risk)

    return {
        "quality_local": quality_local,
        "indicators": ind,
        "assessment": {
            "label": label,
            "score": round(risk, 3)  # 0..1
        }
    }

# ---------- API ----------

@app.post("/api/analyze")
async def analyze_face(image: UploadFile = File(...)):
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    data = await image.read()
    frame = fix_orientation_and_decode(data)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    H, W = frame.shape[:2]
    proc, scale = resize_keep_ratio(frame, max_width=1024)

    # 1) YuNet
    yunet_results_proc = detect_faces_yunet(proc, conf_thresh=0.6)
    if yunet_results_proc:
        results_proc = yunet_results_proc
        detector_used = "yunet"
    else:
        # 2) DNN
        dnn_results_proc = detect_faces_dnn(proc, conf_thresh=0.7)
        if dnn_results_proc:
            results_proc = dnn_results_proc
            detector_used = "dnn"
        else:
            # 3) Haar на CLAHE
            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            results_proc = detect_faces_haar(gray)
            detector_used = "haar"

    # # 1) DNN
    # dnn_results_proc = detect_faces_dnn(proc, conf_thresh=0.7)
    # detector_used = "dnn" if dnn_results_proc else "haar"

    # # 2) Фолбэк: Haar на CLAHE
    # if not dnn_results_proc:
    #     gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     gray = clahe.apply(gray)
    #     results_proc = detect_faces_haar(gray)
    # else:
    #     results_proc = dnn_results_proc

    # Координаты → исходный размер, пост-фильтр
    faces_raw = map_to_original_scale(results_proc, scale)
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness_full = float(gray_full.mean())
    faces = post_filter_faces(frame, faces_raw, brightness=brightness_full)

    faces_count = len(faces)

    # Глобальное качество (на весь кадр)
    blur_full = variance_of_laplacian(gray_full)
    face_area_ratio = 0.0
    if faces_count > 0:
        bx = max(faces, key=lambda r: r["width"] * r["height"])
        face_area_ratio = (bx["width"] * bx["height"]) / (W * H)

    quality = {
        "blur_variance": round(blur_full, 2),
        "brightness_mean": round(brightness_full, 2),
        "face_area_ratio": round(face_area_ratio, 3),
        "ok": (blur_full >= 120.0) and (60.0 <= brightness_full <= 200.0) and (
            (faces_count == 0) or (face_area_ratio >= 0.12)
        ),
        "detector": detector_used,
        "notes": []
    }
    if blur_full < 120.0: quality["notes"].append("low_blur")
    if brightness_full < 60.0: quality["notes"].append("too_dark")
    if brightness_full > 200.0: quality["notes"].append("too_bright")
    if faces_count > 0 and face_area_ratio < 0.12: quality["notes"].append("face_too_small")

    # --- ОЦЕНКА ДЛЯ КАЖДОГО ЛИЦА ---
    faces_out = []
    for f in faces:
        assessed = assess_face(frame, f, (W, H))
        item = {
            "x": int(f["x"]),
            "y": int(f["y"]),
            "width": int(f["width"]),
            "height": int(f["height"]),
            "assessment": assessed["assessment"],   # {label, score}
            "indicators": assessed["indicators"],   # per-face
            "quality_local": assessed["quality_local"]
        }
        if "confidence" in f:
            item["confidence"] = float(f["confidence"])
        faces_out.append(item)

    advice: List[str] = []
    if faces_count == 0:
        advice.append("Лицо не найдено: подойдите ближе к камере и обеспечьте ровное освещение.")
    elif not quality["ok"]:
        advice.append("Качество снимка низкое: переснимите при дневном свете, без фильтров/очков.")
    # Доп. подсказка, если много лиц
    if faces_count >= 4:
        advice.append("Слишком много лиц в кадре — выделение областей может быть неточным.")

    res_obj = {
        "faces_count": faces_count,
        "faces": faces_out,         # теперь с оценкой для каждого лица
        "quality": quality,         # глобальное качество кадра
        "advice": advice,
        "disclaimer": "Предварительная оценка по фото (эвристики). Не является медицинским диагнозом."
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
    <title>FaceHealth Prototype</title>
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
