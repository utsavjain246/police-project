import json
import logging
import os
import re
import shutil
import time

import cv2

from video_reader import VideoReader

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _sanitize_id(video_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video_id))


def processed_dir_for_id(root_dir, video_id):
    safe_id = _sanitize_id(video_id)
    return os.path.join(root_dir, safe_id)


def _format_timestamp(seconds):
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    hours = total_ms // 3600000
    total_ms %= 3600000
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}.{ms:03d}"


def _read_meta(meta_path):
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_meta(meta_path, payload):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _has_any_frames(folder):
    if not os.path.isdir(folder):
        return False
    for name in os.listdir(folder):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            return True
    return False


def ensure_preprocessed(video_id, source_path, root_dir, target_fps=5.0):
    os.makedirs(root_dir, exist_ok=True)
    out_dir = processed_dir_for_id(root_dir, video_id)
    meta_path = os.path.join(out_dir, "meta.json")

    try:
        stat = os.stat(source_path)
    except FileNotFoundError:
        logger.warning("Preprocess source missing: %s", source_path)
        return None

    current_signature = {
        "source_size": stat.st_size,
        "source_mtime": int(stat.st_mtime),
        "target_fps": float(target_fps),
    }
    meta = _read_meta(meta_path)
    if meta and all(meta.get(k) == v for k, v in current_signature.items()) and _has_any_frames(out_dir):
        return out_dir

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    cap = VideoReader(source_path)
    if not cap.isOpened():
        logger.warning("Preprocess open failed: %s", source_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = target_fps
    stride = max(1, int(round(float(fps) / float(target_fps))))

    frame_idx = 0
    saved = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            timestamp = _format_timestamp(frame_idx / float(fps))
            filename = f"frame_{saved:06d}_idx_{frame_idx}_t_{timestamp}.jpg"
            cv2.imwrite(os.path.join(out_dir, filename), frame)
            saved += 1
        frame_idx += 1

    cap.release()

    meta_payload = {
        "source_path": source_path,
        "source_size": stat.st_size,
        "source_mtime": int(stat.st_mtime),
        "source_fps": float(fps),
        "target_fps": float(target_fps),
        "stride": stride,
        "frames": saved,
        "created_at": time.time(),
        "elapsed_sec": round(time.time() - start_time, 3),
    }
    _write_meta(meta_path, meta_payload)
    logger.info("Preprocess complete: %s frames=%s", out_dir, saved)
    return out_dir


def resolve_preprocessed_dir(video_id, source_path, root_dir, target_fps=5.0):
    """Return an existing matching preprocessed directory, or create one if possible."""
    if not source_path or not root_dir:
        return None

    candidates = []
    if video_id:
        candidates.append(video_id)
    basename = os.path.basename(source_path)
    if basename:
        candidates.append(basename)
        candidates.append(f"recorded:{basename}")

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        pre_dir = processed_dir_for_id(root_dir, candidate)
        meta_path = os.path.join(pre_dir, "meta.json")
        meta = _read_meta(meta_path)
        if meta and meta.get("source_path") == source_path and _has_any_frames(pre_dir):
            return pre_dir

    if video_id:
        return ensure_preprocessed(video_id, source_path, root_dir, target_fps)
    return None


def pick_preview_frame(pre_dir):
    if not pre_dir or not os.path.isdir(pre_dir):
        return None
    preview_path = os.path.join(pre_dir, "preview.jpg")
    if os.path.exists(preview_path):
        return preview_path
    frames = [
        name
        for name in sorted(os.listdir(pre_dir))
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
    ]
    if not frames:
        return None
    return os.path.join(pre_dir, frames[0])


def ensure_preview_frame(video_id, source_path, root_dir):
    os.makedirs(root_dir, exist_ok=True)
    out_dir = processed_dir_for_id(root_dir, video_id)
    os.makedirs(out_dir, exist_ok=True)
    preview_path = os.path.join(out_dir, "preview.jpg")
    if os.path.exists(preview_path) and os.path.getsize(preview_path) > 0:
        return preview_path

    cap = VideoReader(source_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    attempts = 0
    while not ret and attempts < 10:
        ret, frame = cap.read()
        attempts += 1
    cap.release()
    if not ret:
        return None
    cv2.imwrite(preview_path, frame)
    return preview_path
