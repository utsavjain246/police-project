
"""
Installation commands

pip install -q "git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/ml-mobileclip"
pip install -q "git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/lvis-api"
pip install -q "git+https://github.com/THU-MIG/yoloe.git"
pip install git+https://github.com/openai/CLIP.git

pip install -q supervision jupyter_bbox_widget
wget -q https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
pip -q install ultralytics supervision opencv-python pandas tqdm

from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg.pt", local_dir='.')
path = hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg-pf.pt", local_dir='.')

"""


import supervision as sv
from ultralytics import YOLOE
from PIL import Image
from tqdm import tqdm

SOURCE_VIDEO_PATH = "/police/12742244_1920_1080_50fps.mp4"
TARGET_VIDEO_PATH = "suitcases_result.mp4"
NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "van"]
PLATE_Names   = ["license plate", "number plate"]

model = YOLOE("yoloe-v8l-seg.pt").cuda()
model.set_classes(NAMES, model.get_text_pe(NAMES))

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# visualize video frames sample in notebook
frames = []
frame_interval = 10

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for index, frame in enumerate(tqdm(frame_generator)):
        results = model.predict(frame, conf=0.1, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        annotated_image = frame.copy()
        annotated_image = sv.ColorAnnotator().annotate(scene=annotated_image, detections=detections)
        annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
        annotated_image = sv.LabelAnnotator().annotate(scene=annotated_image, detections=detections)

        sink.write_frame(annotated_image)

        # visualize video frames sample in notebook
        if index % frame_interval == 0:
            frames.append(annotated_image)

sv.plot_images_grid(frames[:2], grid_size=(2, 1))


import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLOE
from tqdm import tqdm

SOURCE_VIDEO_PATH = "/car-number-plate-video/Traffic Control CCTV.mp4"
TARGET_VIDEO_PATH = "vehicles_with_plate_prep.mp4"

NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "van"]
PLATE_NAMES = ["license plate", "number plate"]

vehicle_model = YOLOE("yoloe-v8l-seg.pt").cuda()
vehicle_model.set_classes(NAMES, vehicle_model.get_text_pe(NAMES))

plate_model = YOLOE("yoloe-v8l-seg.pt").cuda()
plate_model.set_classes(PLATE_NAMES, plate_model.get_text_pe(PLATE_NAMES))


# Tracker: ByteTrack (best "no training" online tracker for bboxes)
tracker = sv.ByteTrack()

# Output dirs
BEST_DIR = "best_plate_per_vehicle"
os.makedirs(BEST_DIR, exist_ok=True)

def clamp_bbox(x1, y1, x2, y2, w, h, pad=0):
    x1 = max(0, int(x1) - pad); y1 = max(0, int(y1) - pad)
    x2 = min(w - 1, int(x2) + pad); y2 = min(h - 1, int(y2) + pad)
    return x1, y1, x2, y2

def crop_masked(frame_bgr, xyxy, mask=None, pad=8):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(*xyxy, w, h, pad=pad)
    crop = frame_bgr[y1:y2, x1:x2].copy()

    if mask is None:
        return crop, None, (x1, y1, x2, y2)

    # mask could be bool/0-1/0-255; make uint8 0/255
    m = mask.astype(np.uint8)
    if m.max() == 1:
        m = m * 255
    m = m[y1:y2, x1:x2]

    masked = cv2.bitwise_and(crop, crop, mask=m)
    return masked, m, (x1, y1, x2, y2)

import numpy as np
import cv2
from PIL import Image

def to_gray_uint8(img):
    """Accept BGR numpy, RGB numpy, or PIL.Image and return 2D np.uint8."""
    if img is None:
        return None

    # PIL -> numpy
    if isinstance(img, Image.Image):
        img = np.array(img)

    # If still not numpy, bail
    if not isinstance(img, np.ndarray):
        return None

    # If object dtype (bad), bail
    if img.dtype == object:
        return None

    # Convert to uint8 safely
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # If BGR/RGB -> gray
    if img.ndim == 3:
        # assume BGR if coming from cv2
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure 2D
    if img.ndim != 2:
        return None

    return img

def enhance_plate_light(plate_bgr):
    if plate_bgr is None or plate_bgr.size == 0:
        return None

    # upscale small plates (good)
    h, w = plate_bgr.shape[:2]
    if h < 96:
        s = 96 / max(h, 1)
        plate_bgr = cv2.resize(plate_bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    # VERY mild denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Mild contrast (CLAHE but softer)
    clahe = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray


def enhance_plate_adaptive(plate_bgr):
    base = enhance_plate_light(plate_bgr)
    if base is None:
        return None, None

    sharpness = float(cv2.Laplacian(base, cv2.CV_64F).var())
    contrast  = float(base.std())

    # If already sharp + decent contrast, keep it simple
    if sharpness > 120 and contrast > 35:
        return base, base  # use same for OCR

    # Otherwise, apply stronger steps
    blur = cv2.GaussianBlur(base, (0, 0), 1.0)
    unsharp = cv2.addWeighted(base, 1.5, blur, -0.5, 0)

    thr = cv2.adaptiveThreshold(
        unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    return base, thr


def laplacian_sharpness(gray):
    gray = to_gray_uint8(gray)
    if gray is None:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def plate_score(plate_bgr, pconf):
    base = enhance_plate_light(plate_bgr)
    if base is None:
        return 0.0, None

    sharp = float(cv2.Laplacian(base, cv2.CV_64F).var())
    contrast = float(base.std())
    h, w = plate_bgr.shape[:2]
    area = h*w

    # Weighted score (tweakable)
    score = (sharp * 0.7 + contrast * 20.0) * (1.0 + 0.7*pconf) * np.log1p(area)
    return float(score), base

class TrackBest:
    __slots__ = ("best_score", "best_plate_bgr", "best_plate_gray", "best_meta", "last_seen")
    def __init__(self):
        self.best_score = -1.0
        self.best_plate_bgr = None
        self.best_plate_gray = None
        self.best_meta = {}
        self.last_seen = -1

track_best = {}  # track_id -> TrackBest

# Tuning knobs
VEH_CONF = 0.25
PLATE_CONF = 0.50

DETECT_STRIDE = 10   # process every frame (use 2 or 3 if slow)
PLATE_STRIDE  = 1   # plate search stride per track update
MAX_MISS_FRAMES = int(video_info.fps * 1.0)  # finalize after ~1s missing

frame_gen = vehicle_model.track(
    source=SOURCE_VIDEO_PATH,
    stream=True,
    conf=VEH_CONF,
    iou=0.5,
    tracker="botsort.yaml",
    persist=True,
    verbose=False
)

box_annot = sv.BoxAnnotator()
label_annot = sv.LabelAnnotator()
color_annot = sv.ColorAnnotator()

saved_tracks = set()

with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame_idx, res in enumerate(tqdm(frame_gen)):
        frame = res.orig_img
        if frame_idx % DETECT_STRIDE != 0:
            continue
        H, W = frame.shape[:2]

        annotated = frame.copy()

        # res.boxes has tracking IDs when persist=True
        if res.boxes is None or res.boxes.xyxy is None or res.boxes.cls is None:
            sink.write_frame(annotated)
            continue

        ids = None
        if res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int)

        xyxy = res.boxes.xyxy.cpu().numpy()
        cls  = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else np.ones(len(xyxy), dtype=float)

        # If IDs are missing for some frames, fall back to per-frame index (not ideal, but avoids crash)
        if ids is None:
            ids = np.arange(len(xyxy), dtype=int)

        # Build Supervision detections just for annotation convenience
        v_tracked = sv.Detections(
            xyxy=xyxy,
            class_id=cls,
            confidence=conf
        )
        # store tracker ids for labels (sv versions differ; we keep it separate)
        labels = []

        for i in range(len(xyxy)):
            tid = int(ids[i])
            c = int(cls[i]) if i < len(cls) else -1
            vname = NAMES[c] if 0 <= c < len(NAMES) else "vehicle"
            labels.append(f"{vname}#{tid}")

            x1, y1, x2, y2 = clamp_bbox(*xyxy[i], W, H, pad=10)
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            # init track store
            st = track_best.get(tid)
            if st is None:
                st = TrackBest()
                track_best[tid] = st
            st.last_seen = frame_idx

            # Plate detection inside vehicle crop
            if frame_idx % PLATE_STRIDE != 0:
                continue

            p_res = plate_model.predict(vehicle_crop, conf=PLATE_CONF, verbose=False)
            p_det = sv.Detections.from_ultralytics(p_res[0])
            if len(p_det) == 0:
                continue

            j = int(np.argmax(p_det.confidence)) if p_det.confidence is not None else 0
            px1, py1, px2, py2 = p_det.xyxy[j]
            pconf = float(p_det.confidence[j]) if p_det.confidence is not None else 0.0

            # map plate box back to full-frame coords
            fx1, fy1, fx2, fy2 = (px1 + x1, py1 + y1, px2 + x1, py2 + y1)
            fx1, fy1, fx2, fy2 = clamp_bbox(fx1, fy1, fx2, fy2, W, H, pad=6)

            plate_bgr = frame[fy1:fy2, fx1:fx2].copy()
            if plate_bgr.size == 0:
                continue

            score, plate_gray = plate_score(plate_bgr, pconf)

            # keep best for this track
            if score > st.best_score:
                st.best_score = score
                st.best_plate_bgr = plate_bgr
                st.best_plate_gray = plate_gray
                st.best_meta = {
                    "frame_idx": frame_idx,
                    "vehicle": vname,
                    "track_id": tid,
                    "plate_conf": pconf,
                    "plate_xyxy": (fx1, fy1, fx2, fy2),
                    "vehicle_xyxy": (x1, y1, x2, y2),
                    "score": score
                }

            # draw plate box
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
            cv2.putText(
                annotated, f"plate {pconf:.2f} score {score:.0f}",
                (fx1, max(0, fy1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )

        # annotate vehicles
        annotated = color_annot.annotate(scene=annotated, detections=v_tracked)
        annotated = box_annot.annotate(scene=annotated, detections=v_tracked)
        annotated = label_annot.annotate(scene=annotated, detections=v_tracked, labels=labels)

        sink.write_frame(annotated)

        # Finalize disappeared tracks (save best plate ONCE per track)
        to_finalize = []
        for tid, st in track_best.items():
            if st.last_seen >= 0 and (frame_idx - st.last_seen) > MAX_MISS_FRAMES:
                to_finalize.append(tid)

        for tid in to_finalize:
            st = track_best[tid]
            if tid in saved_tracks:
                del track_best[tid]
                continue

            meta = st.best_meta or {}
            vname = meta.get("vehicle", "vehicle")
            fidx = meta.get("frame_idx", -1)
            pconf = meta.get("plate_conf", 0.0)
            score = meta.get("score", 0.0)
            out_base = f"{vname}_track{tid}_f{fidx}_p{pconf:.2f}_s{score:.0f}"

            # save RAW + GRAY (better for later OCR comparison)
            if st.best_plate_gray is not None:
                cv2.imwrite(f"{BEST_DIR}/{out_base}_GRAY.png", st.best_plate_gray)

            saved_tracks.add(tid)
            del track_best[tid]

import glob
from PIL import Image
import matplotlib.pyplot as plt

plate_paths = sorted(glob.glob("best_plate_per_vehicle/*.png"))  # change folder if yours differs
print("Found", len(plate_paths), "plate crops")

# show first 20
to_show = plate_paths[:170]

cols = 5
rows = (len(to_show) + cols - 1) // cols
plt.figure(figsize=(15, 3 * rows))

for i, p in enumerate(to_show, 1):
    img = Image.open(p)
    plt.subplot(rows, cols, i)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(p.split("/")[-1][:30])

plt.tight_layout()
plt.show()



import os
import glob
import time
import requests
from tqdm import tqdm

# ================= CONFIG =================
API_URL = "https://unspayed-ruefully-neville.ngrok-free.dev/vlm"
IMG_DIR = "/best_plate_per_vehicle"

MAX_IMAGES = 100
TIMEOUT = (10, 90)
RETRIES = 3
# ==========================================

paths = sorted(
    glob.glob(os.path.join(IMG_DIR, "*.png")) +
    glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
)[:MAX_IMAGES]

session = requests.Session()  # no API key


def read_plate_vlm(path, retries=3):
    for attempt in range(retries):
        try:
            with open(path, "rb") as f:
                files = {
                    "image": (os.path.basename(path), f, "image/jpeg")
                }
                data = {
                    "prompt": (
                        "You are an OCR system.\n"
                        "Task: Read the vehicle license plate text in the image.\n"
                        "Rules:\n"
                        "- Return ONLY the plate number\n"
                        "- Use ONLY uppercase letters and digits\n"
                        "- NO spaces\n"
                        "- NO explanation\n"
                        "- NO description\n"
                        "- If unreadable, return exactly: NONE"
                    )
                }

                r = session.post(
                    API_URL,
                    files=files,
                    data=data,
                    timeout=(10, 90)
                )

            if r.status_code == 200:
                js = r.json()

                # ✅ CORRECT extraction for Qwen-VL style response
                content = (
                    js.get("choices", [{}])[0]
                      .get("message", {})
                      .get("content", "")
                      .strip()
                      .upper()
                )

                # normalize
                import re
                content = re.sub(r"[^A-Z0-9]", "", content)

                if content == "" or content == "NONE":
                    return "", 0.0

                return content, 1.0
        except Exception as e:
            print("ERR:", str(e)[:200])

        time.sleep(0.5 * (2 ** attempt))

    return "", 0.0


# ================= RUN =================
for p in tqdm(paths, desc="VLM plate reading"):
    text, score = read_plate_vlm(p)
    print(f"{os.path.basename(p)} -> {text} ({score:.2f})")




