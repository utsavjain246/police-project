import base64
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from person_video_search import (
    DEFAULT_COOLDOWN_SEC as PERSON_SEARCH_DEFAULT_COOLDOWN_SEC,
    DEFAULT_MAX_MATCHES as PERSON_SEARCH_DEFAULT_MAX_MATCHES,
    DEFAULT_MIN_CONFIDENCE as PERSON_SEARCH_DEFAULT_MIN_CONFIDENCE,
    DEFAULT_OUTPUT_DIR as PERSON_SEARCH_DEFAULT_OUTPUT_DIR,
    DEFAULT_SCAN_FPS as PERSON_SEARCH_DEFAULT_SCAN_FPS,
    PersonVideoSearch,
)
from db_cache import ForensicDB
from preprocess import resolve_preprocessed_dir
from recorded_orchestrator import RecordedOrchestrator
from video_reader import VideoReader
from llm_verifier import LLMVerifier
import logging
import config


config.configure_logging()
app = FastAPI(title="Forensic Analysis API")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = RecordedOrchestrator()
verifier = LLMVerifier()


jobs = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=int(os.getenv("JOB_WORKERS", "3")))
TERMINAL_JOB_STATUSES = {"completed", "failed", "canceled"}
SUPPORTED_VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".3gp",
}
MAX_THUMBNAILS = max(1, int(os.getenv("MAX_THUMBNAILS", "20")))
JOB_RETENTION_SECONDS = max(60, int(os.getenv("JOB_RETENTION_SECONDS", "1800")))
UPLOAD_COPY_CHUNK_BYTES = 4 * 1024 * 1024
FACE_SEARCH_FPS = float(os.getenv("FACE_SEARCH_FPS", str(PERSON_SEARCH_DEFAULT_SCAN_FPS)))
FACE_SEARCH_MIN_CONF = float(os.getenv("FACE_SEARCH_MIN_CONF", str(PERSON_SEARCH_DEFAULT_MIN_CONFIDENCE)))
FACE_SEARCH_MAX_MATCHES = int(os.getenv("FACE_SEARCH_MAX_MATCHES", str(PERSON_SEARCH_DEFAULT_MAX_MATCHES)))
FACE_SEARCH_COOLDOWN_SEC = float(os.getenv("FACE_SEARCH_COOLDOWN_SEC", str(PERSON_SEARCH_DEFAULT_COOLDOWN_SEC)))
FACE_SEARCH_OUTPUT_DIR = os.getenv("FACE_SEARCH_OUTPUT_DIR", PERSON_SEARCH_DEFAULT_OUTPUT_DIR)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_STORE_DIR = os.getenv("VIDEO_STORE_DIR", os.path.join(BACKEND_DIR, "video_store"))
RECORDED_CLIPS_DIR = os.getenv("RECORDED_CLIPS_DIR", VIDEO_STORE_DIR)
PROCESSED_FRAMES_DIR = os.getenv("PROCESSED_FRAMES_DIR", os.path.join(BACKEND_DIR, "video_processed"))
PREPROCESS_FPS = float(os.getenv("PREPROCESS_FPS", "5"))
LPR_EVIDENCE_DIR = os.getenv("LPR_EVIDENCE_DIR", os.path.join(BACKEND_DIR, "evidence", "lpr"))
FORENSIC_DB_PATH = os.getenv("FORENSIC_DB_PATH", os.path.join(BACKEND_DIR, "forensic_cache.sqlite3"))
db = ForensicDB(FORENSIC_DB_PATH, VIDEO_STORE_DIR)
db.init_db()

def _ensure_storage_ready():
    db.ensure_ready()


def _serve_cached_job(*, video_id, query_text, request_received_at, report, source_clip=None):
    now = time.time()
    cached_report = json.loads(json.dumps(report)) if isinstance(report, dict) else report
    timings = _build_job_timings(
        {
            "request_received_at": request_received_at,
            "upload_saved_at": now,
            "created_at": now,
        },
        now,
        now,
    )
    if isinstance(cached_report, dict):
        cached_report["timings"] = timings
        cached_report["cache_hit"] = True
        cached_report.setdefault("query", query_text)

    job_id = uuid4().hex
    set_job(
        job_id,
        status="completed",
        progress=100,
        message="Complete (cache hit)",
        created_at=now,
        request_received_at=request_received_at,
        upload_saved_at=now,
        started_at=now,
        completed_at=now,
        timings=timings,
        result=cached_report,
        cache_hit=False,
        video_id=video_id,
        source_clip=source_clip,
    )
    return {"job_id": job_id, "video_id": video_id, "cached": True}


def set_job(job_id, **updates):
    with jobs_lock:
        current = jobs.get(job_id, {})
        current.update(updates)
        jobs[job_id] = current
        logger.debug("Job updated: id=%s updates=%s", job_id, updates)


def get_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        logger.debug("Job fetched: id=%s exists=%s", job_id, bool(job))
        return job


class JobCanceledError(RuntimeError):
    pass


def _job_cancel_requested(job_id):
    job = get_job(job_id) or {}
    return bool(job.get("cancel_requested"))


def _mark_job_canceled(job_id, message="Canceled by user"):
    job = get_job(job_id) or {}
    if job.get("status") == "canceled":
        return job
    completed_at = time.time()
    timings = _build_job_timings(job, job.get("started_at"), completed_at)
    progress = int(job.get("progress") or 0)
    set_job(
        job_id,
        status="canceled",
        message=message,
        progress=min(100, max(0, progress)),
        completed_at=completed_at,
        timings=timings,
        cancel_requested=False,
    )
    return get_job(job_id) or {}


def _raise_if_job_canceled(job_id, message="Canceled by user"):
    if _job_cancel_requested(job_id):
        _mark_job_canceled(job_id, message)
        raise JobCanceledError(message)


def _duration_ms(start_ts, end_ts):
    if start_ts is None or end_ts is None:
        return None
    try:
        elapsed_ms = int(round((float(end_ts) - float(start_ts)) * 1000.0))
    except (TypeError, ValueError):
        return None
    return max(0, elapsed_ms)


def _build_job_timings(job, started_at, completed_at):
    job = job or {}
    request_received_at = job.get("request_received_at")
    upload_saved_at = job.get("upload_saved_at")
    queued_at = job.get("created_at")

    timings = {
        "request_received_at": request_received_at,
        "upload_saved_at": upload_saved_at,
        "queued_at": queued_at,
        "started_at": started_at,
        "completed_at": completed_at,
    }

    timings["request_to_upload_ms"] = _duration_ms(request_received_at, upload_saved_at)
    timings["queue_wait_ms"] = _duration_ms(queued_at, started_at)
    timings["processing_ms"] = _duration_ms(started_at, completed_at)
    timings["post_upload_to_complete_ms"] = _duration_ms(upload_saved_at, completed_at)
    timings["job_lifecycle_ms"] = _duration_ms(queued_at, completed_at)
    timings["end_to_end_request_ms"] = _duration_ms(request_received_at, completed_at)

    return timings


def _encode_frame(frame, bbox=None):
    import cv2

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        logger.warning("Failed to encode thumbnail frame")
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _draw_bbox(frame, bbox):
    import cv2

    if frame is None or bbox is None:
        return frame
    try:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    except Exception:
        return frame
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


def _decode_thumbnail_to_image(thumbnail):
    if not thumbnail or not isinstance(thumbnail, str):
        return None
    if not thumbnail.startswith("data:image"):
        return None
    try:
        _, b64 = thumbnail.split(",", 1)
        data = base64.b64decode(b64)
    except Exception:
        return None
    try:
        import cv2
        import numpy as np

        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _overlay_bbox_on_thumbnail(thumbnail, bbox):
    image = _decode_thumbnail_to_image(thumbnail)
    if image is None:
        return thumbnail
    image = _draw_bbox(image, bbox)
    return _encode_frame(image)




def _add_thumbnails(report, video_path):
    import cv2

    if not report or not report.get("results"):
        logger.debug("No results for thumbnails")
        return report

    cap = VideoReader(video_path)
    if not cap.isOpened():
        logger.warning("Video not readable for thumbnails: %s", video_path)
        return report

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Handle both legacy list results and new dict format {"summary_stats": {}, "events": []}
    raw_results = report["results"]
    if isinstance(raw_results, dict) and "events" in raw_results:
        items = raw_results["events"]
    else:
        items = raw_results
        
    logger.debug("Thumbnail generation fps=%s items=%s", fps, len(items))
    missing_indexes = [i for i, item in enumerate(items) if not item.get("thumbnail")]
    if not missing_indexes:
        logger.debug("All items already have thumbnails")
        cap.release()
        return report

    if len(missing_indexes) == len(items):
        selected_indexes = set(range(len(items)))
    elif len(missing_indexes) > MAX_THUMBNAILS:
        step = len(missing_indexes) / float(MAX_THUMBNAILS)
        selected_indexes = {missing_indexes[min(len(missing_indexes) - 1, int(i * step))] for i in range(MAX_THUMBNAILS)}
    else:
        selected_indexes = set(missing_indexes)

    added = 0
    for idx, item in enumerate(items):
        if idx not in selected_indexes:
            continue
        if item.get("thumbnail"):
            continue
        time_sec = item.get("time_sec")
        if time_sec is None and item.get("timestamp") is not None:
            time_sec = item.get("timestamp")

        if time_sec is None:
            continue

        # Seek by time for best compatibility
        cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
        ret, frame = cap.read()
        if not ret:
            logger.debug("Thumbnail read failed at time_sec=%s frame=%s", time_sec, item.get("frame"))
            # Fallback: try frame index if provided
            frame_idx = item.get("frame")
            if frame_idx is not None and fps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                if not ret:
                    logger.debug("Thumbnail read failed at frame_idx=%s", frame_idx)
                    continue
            else:
                continue

        frame = _draw_bbox(frame, item.get("bbox"))
        thumb = _encode_frame(frame)
        if thumb:
            item["thumbnail"] = thumb
            added += 1

    cap.release()
    logger.info("Thumbnail ensure complete: added=%s missing=%s", added, len(missing_indexes))
    return report


def _extract_single_thumbnail(video_path, *, time_sec=None, frame_idx=None):
    import cv2

    if time_sec is None and frame_idx is None:
        return None

    cap = VideoReader(video_path)
    if not cap.isOpened():
        logger.warning("Video not readable for single thumbnail: %s", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret = False
    frame = None
    try:
        if time_sec is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
            ret, frame = cap.read()

        if not ret and frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

        if not ret and frame_idx is not None and fps:
            cap.set(cv2.CAP_PROP_POS_MSEC, (int(frame_idx) / float(fps)) * 1000.0)
            ret, frame = cap.read()

        if not ret:
            return None
        frame = _draw_bbox(frame, None)
        return _encode_frame(frame)
    finally:
        cap.release()


def _is_video_readable(path):
    import cv2

    cap = VideoReader(path)
    if not cap.isOpened():
        logger.warning("Video not opened: %s", path)
        cap.release()
        return False
    ret, _ = cap.read()
    cap.release()
    logger.debug("Video readable=%s path=%s", bool(ret), path)
    return bool(ret)


def _transcode_to_mp4(src_path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.error("FFmpeg not found in PATH")
        raise RuntimeError(
            "FFmpeg is required to process this video format. "
            "Install FFmpeg or upload a standard .mp4 file."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        dst_path = tmp_file.name

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        src_path,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        dst_path,
    ]

    logger.info("Transcoding with ffmpeg: src=%s", src_path)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        logger.error("FFmpeg failed: %s", result.stderr.strip())
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.strip()}")

    logger.info("Transcode complete: dst=%s", dst_path)
    return dst_path


def _prepare_video_for_analysis(video_path):
    _, ext = os.path.splitext(video_path)
    ext = ext.lower()
    if ext and ext not in SUPPORTED_VIDEO_EXTS:
        logger.warning("Unsupported video format for analysis: %s", ext)
        raise ValueError(f"Unsupported video format: {ext}")

    if _is_video_readable(video_path):
        logger.debug("Video is readable; no transcode needed: %s", video_path)
        return video_path

    logger.info("Video not readable; transcoding: %s", video_path)
    return _transcode_to_mp4(video_path)


def _prepare_preview_video(video_path):
    _, ext = os.path.splitext(video_path)
    ext = ext.lower()
    if ext != ".mp4":
        logger.info("Preview requires mp4; transcoding: %s", video_path)
        return _transcode_to_mp4(video_path)
    if _is_video_readable(video_path):
        logger.debug("Preview video readable: %s", video_path)
        return video_path
    logger.info("Preview video not readable; transcoding: %s", video_path)
    return _transcode_to_mp4(video_path)


def _cleanup_file(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.debug("Cleaned up file: %s", path)
    except OSError:
        logger.warning("Failed to cleanup file: %s", path)
        pass


def _cleanup_dir(path):
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            logger.debug("Cleaned up directory: %s", path)
    except OSError:
        logger.warning("Failed to cleanup directory: %s", path)
        pass


def _maybe_cleanup_job(job_id, job):
    if not job:
        return False
    status = job.get("status")
    completed_at = job.get("completed_at")
    if status not in TERMINAL_JOB_STATUSES or not completed_at:
        return False
    if time.time() - completed_at < JOB_RETENTION_SECONDS:
        return False

    for key in ("upload_path", "preview_path", "analysis_path", "reference_path", "report_path"):
        _cleanup_file(job.get(key))
    _cleanup_dir(job.get("evidence_dir"))

    with jobs_lock:
        jobs.pop(job_id, None)
    logger.info("Job expired and cleaned: id=%s", job_id)
    return True


def _save_upload_to_temp(video: UploadFile, suffix: str):
    save_start = time.time()
    try:
        video.file.seek(0, os.SEEK_END)
        size = video.file.tell()
        video.file.seek(0)
    except Exception:
        size = None

    logger.info(
        "Upload received: name=%s content_type=%s size=%s",
        video.filename,
        getattr(video, "content_type", None),
        size,
    )

    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded video is empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name

    try:
        with open(tmp_path, "wb") as out_file:
            shutil.copyfileobj(video.file, out_file, length=UPLOAD_COPY_CHUNK_BYTES)
        elapsed = time.time() - save_start
        logger.info("Upload saved: %s in %.2fs", tmp_path, elapsed)
    except Exception as exc:
        _cleanup_file(tmp_path)
        logger.exception("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}")

    return tmp_path


def _save_upload_image_to_temp(image: UploadFile):
    if not image or not image.filename:
        raise HTTPException(status_code=400, detail="Image file is required")
    _, ext = os.path.splitext(image.filename)
    suffix = ext if ext else ".png"
    return _save_upload_to_temp(image, suffix)


def _resolve_recorded_clip(clip_id: str):
    if not clip_id or not isinstance(clip_id, str):
        raise HTTPException(status_code=400, detail="clip_id is required")

    normalized = os.path.basename(clip_id.strip())
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid clip_id")

    clip_path = os.path.join(RECORDED_CLIPS_DIR, normalized)
    if not os.path.isfile(clip_path):
        raise HTTPException(status_code=404, detail="Clip not found")

    _, ext = os.path.splitext(normalized)
    if ext.lower() not in SUPPORTED_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported clip format: {ext.lower()}")

    return clip_path, normalized


def _list_recorded_clips():
    if not os.path.isdir(RECORDED_CLIPS_DIR):
        return []

    clips = []
    for name in sorted(os.listdir(RECORDED_CLIPS_DIR)):
        clip_path = os.path.join(RECORDED_CLIPS_DIR, name)
        if not os.path.isfile(clip_path):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() not in SUPPORTED_VIDEO_EXTS:
            continue
        stat = os.stat(clip_path)
        clips.append(
            {
                "id": name,
                "name": name,
                "size_bytes": stat.st_size,
                "updated_at": int(stat.st_mtime),
                "stream_url": f"/recorded/clips/{name}",
            }
        )
    return clips


def run_analysis_job(
    job_id,
    video_path,
    query,
    include_thumbnails,
    video_id=None,
    normalized_query=None,
    preprocess_source_path=None,
    reference_image=None,
):
    logger.info("Job start: id=%s query=%s include_thumbnails=%s", job_id, query, include_thumbnails)
    job_started_at = time.time()
    try:
        _raise_if_job_canceled(job_id, "Canceled before processing started")
        set_job(job_id, status="running", progress=5, message="Preparing video", started_at=job_started_at)
        cancel_check = lambda: _job_cancel_requested(job_id)
        analysis_source = video_path
        pre_dir = None
        if video_id and preprocess_source_path:
            try:
                pre_dir = resolve_preprocessed_dir(
                    video_id,
                    preprocess_source_path,
                    PROCESSED_FRAMES_DIR,
                    PREPROCESS_FPS,
                )
            except Exception as exc:
                logger.warning("Preprocess failed for video_id=%s: %s", video_id, exc)
        _raise_if_job_canceled(job_id, "Canceled while preparing video")
        prepared_path = _prepare_video_for_analysis(analysis_source)
        set_job(job_id, analysis_path=prepared_path)

        set_job(job_id, progress=10, message="Analyzing")
        lpr_evidence_dir = os.path.join(LPR_EVIDENCE_DIR, job_id)
        report = orchestrator.process(
            prepared_path,
            query,
            lpr_evidence_dir=lpr_evidence_dir,
            reference_image=reference_image,
            cancel_check=cancel_check,
            preprocessed_dir=pre_dir,
        )
        _raise_if_job_canceled(job_id, "Canceled during analysis")
        if isinstance(report, dict) and report.get("tool_used") == "LPR":
            set_job(job_id, evidence_dir=lpr_evidence_dir)

        set_job(job_id, progress=85, message="Ensuring thumbnails")
        if report.get("response_type") != "text":
            report = _add_thumbnails(report, prepared_path)
        _raise_if_job_canceled(job_id, "Canceled while preparing evidence")

        # Persist pre-verification report to disk
        try:
            evidence_dir = os.path.join(BACKEND_DIR, "evidence", "api_jobs", job_id)
            os.makedirs(evidence_dir, exist_ok=True)
            pre_verify_path = os.path.join(evidence_dir, "report_pre_verify.json")
            with open(pre_verify_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            set_job(job_id, report_path=pre_verify_path, evidence_dir=evidence_dir)
        except Exception as exc:
            logger.warning("Failed to save pre-verify report: %s", exc)

        set_job(job_id, progress=90, message="Verifying results")
        if report.get("response_type") != "text":
            raw_results = report.get("results")
            
            # Unpack events list if results is a dictionary (like the new LPR output)
            if isinstance(raw_results, dict) and "events" in raw_results:
                events_list = raw_results["events"]
            else:
                events_list = raw_results or []
                
            import asyncio
            verified = asyncio.run(verifier.verify_results(report.get("query", query), events_list))
            
            # Repack events back into dictionary or list
            if isinstance(raw_results, dict) and "events" in raw_results:
                raw_results["events"] = verified
                report["results"] = raw_results
            else:
                report["results"] = verified

            report["events_found"] = len(verified)
            meta = {
                "enabled": bool(getattr(verifier, "enabled", True)),
                "total": len(events_list),
                "kept": len(verified),
                "dropped": max(0, len(events_list) - len(verified)),
            }
            report["verification"] = meta
            logger.info("Verification meta: %s", meta)

            # Overlay bbox on kept results only
            for event in verified:
                if event.get("thumbnail") and event.get("bbox"):
                    event["thumbnail"] = _overlay_bbox_on_thumbnail(
                        event.get("thumbnail"),
                        event.get("bbox"),
                    )
        _raise_if_job_canceled(job_id, "Canceled before completion")

        completed_at = time.time()
        timings = _build_job_timings(get_job(job_id), job_started_at, completed_at)
        if isinstance(report, dict):
            report["timings"] = timings
            if video_id:
                report["video_id"] = video_id
        set_job(
            job_id,
            status="completed",
            progress=100,
            message="Complete",
            result=report,
            completed_at=completed_at,
            timings=timings,
        )
        if video_id and isinstance(report, dict):
            try:
                query_norm = normalized_query or db.normalize_query(query)
                db.cache_report(video_id, query_norm, query, report)
            except Exception as exc:
                logger.warning("Failed to write analysis cache for video_id=%s: %s", video_id, exc)
        logger.info("Job complete: id=%s", job_id)
    except JobCanceledError:
        logger.info("Job canceled: id=%s", job_id)
    except Exception as exc:
        logger.exception("Job failed: id=%s error=%s", job_id, exc)
        completed_at = time.time()
        timings = _build_job_timings(get_job(job_id), job_started_at, completed_at)
        set_job(
            job_id,
            status="failed",
            progress=100,
            error=str(exc),
            completed_at=completed_at,
            timings=timings,
        )


def run_face_search_job(
    job_id,
    video_path,
    reference_path,
    query,
    include_thumbnails,
    scan_fps,
    min_confidence,
    max_matches,
    source_clip_path=None,
):
    logger.info(
        "Face search job start: id=%s scan_fps=%s min_conf=%s max_matches=%s",
        job_id,
        scan_fps,
        min_confidence,
        max_matches,
    )
    job_started_at = time.time()
    try:
        _raise_if_job_canceled(job_id, "Canceled before processing started")
        set_job(job_id, status="running", progress=5, message="Preparing video", started_at=job_started_at)
        prepared_path = _prepare_video_for_analysis(video_path)
        _raise_if_job_canceled(job_id, "Canceled while preparing video")
        set_job(job_id, analysis_path=prepared_path, progress=10, message="Scanning frames")

        # Resolve preprocessed frames directory if available
        pre_dir = None
        if source_clip_path:
            try:
                clip_name = os.path.basename(source_clip_path)
                video_id = f"recorded:{clip_name}" if clip_name else None
                pre_dir = resolve_preprocessed_dir(
                    video_id, source_clip_path, PROCESSED_FRAMES_DIR, PREPROCESS_FPS
                )
                if pre_dir:
                    logger.info("Using preprocessed frames for face search: %s", pre_dir)
            except Exception as exc:
                logger.warning("Failed to resolve preprocessed dir: %s", exc)
        
        # NOTE: For face search, PersonVideoSearch already handles preprocessed_dir separately.
        # We pass prepared_path (original video or transcode) and pre_dir.

        engine = PersonVideoSearch()
        job_output_dir = os.path.join(FACE_SEARCH_OUTPUT_DIR, "api_jobs", job_id)
        report = engine.search(
            video_path=prepared_path,
            reference_image_path=reference_path,
            query_text=query,
            scan_fps=scan_fps,
            min_confidence=min_confidence,
            max_matches=max_matches,
            cooldown_sec=FACE_SEARCH_COOLDOWN_SEC,
            output_dir=job_output_dir,
            show_progress=False,
            preprocessed_dir=pre_dir,
        )
        _raise_if_job_canceled(job_id, "Canceled during face search")

        set_job(job_id, progress=90, message="Preparing response")
        results = report.get("results") or []
        if include_thumbnails:
            for item in results:
                thumb = _extract_single_thumbnail(
                    prepared_path,
                    time_sec=item.get("time_sec"),
                    frame_idx=item.get("frame"),
                )
                if thumb:
                    item["thumbnail"] = thumb
        _raise_if_job_canceled(job_id, "Canceled while preparing response")

        report["query"] = query
        report["video"] = prepared_path
        report["reference_image"] = reference_path
        report["tool_used"] = "FACE_SEARCH"
        report["events_found"] = len(results)

        completed_at = time.time()
        timings = _build_job_timings(get_job(job_id), job_started_at, completed_at)
        report["timings"] = timings

        set_job(
            job_id,
            status="completed",
            progress=100,
            message="Complete",
            result=report,
            completed_at=completed_at,
            timings=timings,
            tool_used=report.get("tool_used"),
            evidence_dir=report.get("output_dir"),
            report_path=report.get("report_path"),
        )
        logger.info("Face search job complete: id=%s results=%s", job_id, len(results))
    except JobCanceledError:
        logger.info("Face search job canceled: id=%s", job_id)
    except Exception as exc:
        logger.exception("Face search job failed: id=%s error=%s", job_id, exc)
        completed_at = time.time()
        timings = _build_job_timings(get_job(job_id), job_started_at, completed_at)
        set_job(
            job_id,
            status="failed",
            progress=100,
            error=str(exc),
            completed_at=completed_at,
            timings=timings,
        )


@app.get("/health")
def health():
    logger.debug("Health check")
    return {"status": "ok"}


@app.get("/recorded/clips")
def recorded_clips():
    clips = _list_recorded_clips()
    return {"clips": clips}


@app.get("/recorded/clips/{clip_name}")
def recorded_clip_stream(clip_name: str, thumb: int = Query(0)):
    clip_path, normalized = _resolve_recorded_clip(clip_name)
    if thumb:
        try:
            cap = VideoReader(clip_path)
            if not cap.isOpened():
                raise HTTPException(status_code=404, detail="Clip not readable")
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise HTTPException(status_code=404, detail="Thumbnail not available")
            encoded = _encode_frame(frame)
            if not encoded:
                raise HTTPException(status_code=404, detail="Thumbnail not available")
            _, b64 = encoded.split(",", 1)
            data = base64.b64decode(b64)
            return Response(content=data, media_type="image/jpeg")
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Thumbnail generate failed for %s: %s", clip_name, exc)
            raise HTTPException(status_code=500, detail="Thumbnail generation failed")
    return FileResponse(clip_path, media_type="video/mp4", filename=normalized)


@app.post("/analyze_recorded")
async def analyze_recorded(
    clip_id: str = Form(...),
    query: str = Form(...),
    include_thumbnails: bool = Form(True),
    reference_image: UploadFile = File(None),
):
    logger.info("Analyze recorded request received")
    request_received_at = time.time()
    _ensure_storage_ready()
    query_text = (query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query is required")
    normalized_query = db.normalize_query(query_text)
    
    ref_image_path = None
    if reference_image and getattr(reference_image, "filename", None):
        ref_image_path = _save_upload_image_to_temp(reference_image)

    clip_path, clip_name = _resolve_recorded_clip(clip_id)
    clip_stat = os.stat(clip_path)
    recorded_video_id = f"recorded:{clip_name}"
    recorded_checksum = f"{clip_stat.st_size}:{int(clip_stat.st_mtime)}"
    db.upsert_video(
        recorded_video_id,
        source_type="recorded",
        source_ref=clip_name,
        storage_path=clip_path,
        size_bytes=clip_stat.st_size,
        checksum_sha256=recorded_checksum,
    )

    cached_report = db.get_cached_report(recorded_video_id, normalized_query)
    if cached_report:
        logger.info("Recorded cache hit: clip=%s", clip_name)
        return _serve_cached_job(
            video_id=recorded_video_id,
            query_text=query_text,
            request_received_at=request_received_at,
            report=cached_report,
            source_clip=clip_name,
        )

    _, ext = os.path.splitext(clip_name)
    suffix = ext if ext else ".mp4"
    tmp_path = db.stage_video_copy(clip_path, suffix)

    upload_saved_at = time.time()
    queued_at = time.time()

    job_id = uuid4().hex
    set_job(
        job_id,
        status="queued",
        progress=0,
        message="Queued",
        created_at=queued_at,
        request_received_at=request_received_at,
        upload_saved_at=upload_saved_at,
        upload_path=tmp_path,
        preview_path=None,
        analysis_path=None,
        completed_at=None,
        timings=None,
        source_clip=clip_name,
        video_id=recorded_video_id,
        normalized_query=normalized_query,
    )

    executor.submit(
        run_analysis_job,
        job_id,
        tmp_path,
        query_text,
        include_thumbnails,
        recorded_video_id,
        normalized_query,
        clip_path,
        ref_image_path,
    )
    logger.info("Recorded clip job queued: id=%s clip=%s", job_id, clip_name)
    return {"job_id": job_id, "video_id": recorded_video_id, "cached": False}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(None),
    query: str = Form(...),
    include_thumbnails: bool = Form(True),
    video_id: str = Form(None),
    reference_image: UploadFile = File(None),
):
    logger.info("Analyze request received")
    request_received_at = time.time()
    _ensure_storage_ready()
    query_text = (query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query is required")
    normalized_query = db.normalize_query(query_text)
    
    ref_image_path = None
    if reference_image and getattr(reference_image, "filename", None):
        ref_image_path = _save_upload_image_to_temp(reference_image)

    effective_video_id = db.validate_video_id(video_id) if video_id else None
    if effective_video_id:
        cached_report = db.get_cached_report(effective_video_id, normalized_query)
        if cached_report:
            logger.info("Analyze cache hit: video_id=%s", effective_video_id)
            return _serve_cached_job(
                video_id=effective_video_id,
                query_text=query_text,
                request_received_at=request_received_at,
                report=cached_report,
            )

    if not video and not effective_video_id:
        raise HTTPException(status_code=400, detail="Video file or video_id is required")
    if video and not getattr(video, "filename", None) and not effective_video_id:
        raise HTTPException(status_code=400, detail="Video file is required")

    tmp_path = None
    if video and video.filename:
        _, ext = os.path.splitext(video.filename)
        if ext and ext.lower() not in SUPPORTED_VIDEO_EXTS:
            logger.warning("Unsupported upload extension: %s", ext.lower())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format: {ext.lower()}",
            )
        suffix = ext if ext else ".mp4"
        tmp_path = _save_upload_to_temp(video, suffix)

        file_hash = db.sha256_file(tmp_path)
        if not effective_video_id:
            effective_video_id = f"upload:{file_hash[:16]}"
        stored_path = db.persist_video_copy(effective_video_id, tmp_path, suffix)
        db.upsert_video(
            effective_video_id,
            source_type="upload",
            source_ref=video.filename,
            storage_path=stored_path,
            size_bytes=os.path.getsize(stored_path),
            checksum_sha256=file_hash,
        )

        cached_report = db.get_cached_report(effective_video_id, normalized_query)
        if cached_report:
            logger.info("Analyze cache hit after upload: video_id=%s", effective_video_id)
            _cleanup_file(tmp_path)
            return _serve_cached_job(
                video_id=effective_video_id,
                query_text=query_text,
                request_received_at=request_received_at,
                report=cached_report,
            )
    else:
        video_row = db.get_video(effective_video_id)
        if not video_row:
            raise HTTPException(status_code=404, detail="video_id not found in video registry")
        source_path = video_row.get("storage_path")
        if not source_path or not os.path.exists(source_path):
            raise HTTPException(status_code=404, detail="Stored video not found for video_id")
        _, ext = os.path.splitext(source_path)
        suffix = ext if ext else ".mp4"
        tmp_path = db.stage_video_copy(source_path, suffix)

    upload_saved_at = time.time()
    queued_at = time.time()

    job_id = uuid4().hex
    set_job(
        job_id,
        status="queued",
        progress=0,
        message="Queued",
        created_at=queued_at,
        request_received_at=request_received_at,
        upload_saved_at=upload_saved_at,
        upload_path=tmp_path,
        preview_path=None,
        analysis_path=None,
        completed_at=None,
        timings=None,
        video_id=effective_video_id,
        normalized_query=normalized_query,
    )

    executor.submit(
        run_analysis_job,
        job_id,
        tmp_path,
        query_text,
        include_thumbnails,
        effective_video_id,
        normalized_query,
        stored_path if video and video.filename else source_path,
        ref_image_path,
    )
    logger.info("Job queued: id=%s", job_id)
    return {"job_id": job_id, "video_id": effective_video_id, "cached": False}


@app.post("/preview")
async def preview(video: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    logger.info("Preview request received")
    if not video or not video.filename:
        raise HTTPException(status_code=400, detail="Video file is required")

    _, ext = os.path.splitext(video.filename)
    suffix = ext if ext else ".mp4"

    tmp_path = _save_upload_to_temp(video, suffix)

    try:
        preview_path = _prepare_preview_video(tmp_path)
    except Exception as exc:
        _cleanup_file(tmp_path)
        logger.exception("Preview failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    if background_tasks is None:
        background_tasks = BackgroundTasks()
    background_tasks.add_task(_cleanup_file, tmp_path)
    if preview_path != tmp_path:
        background_tasks.add_task(_cleanup_file, preview_path)

    return FileResponse(
        preview_path,
        media_type="video/mp4",
        filename="preview.mp4",
        background=background_tasks,
    )


@app.post("/search_face")
async def search_face(
    video: UploadFile = File(None),
    photo: UploadFile = File(...),
    clip_id: str = Form(None),
    query: str = Form(""),
    include_thumbnails: bool = Form(True),
    scan_fps: float = Form(FACE_SEARCH_FPS),
    min_confidence: float = Form(FACE_SEARCH_MIN_CONF),
    max_matches: int = Form(FACE_SEARCH_MAX_MATCHES),
):
    logger.info("Face search request received")
    request_received_at = time.time()
    db.ensure_ready()
    if clip_id:
        clip_path, clip_name = _resolve_recorded_clip(clip_id)
        _, ext = os.path.splitext(clip_name)
        suffix = ext if ext else ".mp4"
        tmp_video_path = db.stage_video_copy(clip_path, suffix)
        source_clip = clip_name
    else:
        if not video or not video.filename:
            raise HTTPException(status_code=400, detail="Video file is required")
        _, ext = os.path.splitext(video.filename)
        if ext and ext.lower() not in SUPPORTED_VIDEO_EXTS:
            logger.warning("Unsupported upload extension: %s", ext.lower())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format: {ext.lower()}",
            )
        suffix = ext if ext else ".mp4"
        tmp_video_path = _save_upload_to_temp(video, suffix)
        source_clip = None

    tmp_photo_path = _save_upload_image_to_temp(photo)
    upload_saved_at = time.time()
    queued_at = time.time()

    job_id = uuid4().hex
    set_job(
        job_id,
        status="queued",
        progress=0,
        message="Queued",
        created_at=queued_at,
        request_received_at=request_received_at,
        upload_saved_at=upload_saved_at,
        upload_path=tmp_video_path,
        preview_path=None,
        analysis_path=None,
        reference_path=tmp_photo_path,
        evidence_dir=None,
        report_path=None,
        tool_used="FACE_SEARCH",
        source_clip=source_clip,
        completed_at=None,
        timings=None,
    )

    executor.submit(
        run_face_search_job,
        job_id,
        tmp_video_path,
        tmp_photo_path,
        query.strip(),
        include_thumbnails,
        scan_fps,
        min_confidence,
        max_matches,
        source_clip_path=clip_path if clip_id else None,
    )
    logger.info("Face search job queued: id=%s", job_id)
    return {"job_id": job_id}


@app.get("/preview/{job_id}")
def preview_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    preview_path = job.get("preview_path")
    if preview_path and os.path.exists(preview_path):
        logger.debug("Serving cached preview: id=%s path=%s", job_id, preview_path)
        return FileResponse(preview_path, media_type="video/mp4", filename="preview.mp4")

    source_path = job.get("analysis_path") or job.get("upload_path")
    if not source_path or not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Preview not available")

    try:
        preview_path = _prepare_preview_video(source_path)
    except Exception as exc:
        logger.exception("Preview generation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    set_job(job_id, preview_path=preview_path)
    logger.info("Preview generated: id=%s path=%s", job_id, preview_path)
    return FileResponse(preview_path, media_type="video/mp4", filename="preview.mp4")


@app.get("/jobs/{job_id}/thumbnail")
def job_thumbnail(
    job_id: str,
    result_index: int = Query(None),
    time_sec: float = Query(None),
    frame: int = Query(None),
):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if _maybe_cleanup_job(job_id, job):
        raise HTTPException(status_code=404, detail="Job expired")

    source_path = job.get("analysis_path") or job.get("upload_path")
    if not source_path or not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Video source unavailable")

    local_time = time_sec
    local_frame = frame

    if result_index is not None:
        result = (job.get("result") or {})
        items = result.get("results") or []
        if result_index < 0 or result_index >= len(items):
            raise HTTPException(status_code=400, detail="Invalid result_index")
        item = items[result_index]
        if local_time is None:
            local_time = item.get("time_sec", item.get("timestamp"))
        if local_frame is None:
            local_frame = item.get("frame")

    thumbnail = _extract_single_thumbnail(
        source_path,
        time_sec=local_time,
        frame_idx=local_frame,
    )
    if not thumbnail:
        raise HTTPException(status_code=404, detail="Thumbnail not available")

    return {
        "job_id": job_id,
        "result_index": result_index,
        "time_sec": local_time,
        "frame": local_frame,
        "thumbnail": thumbnail,
    }


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if _maybe_cleanup_job(job_id, job):
        raise HTTPException(status_code=404, detail="Job expired")

    status = job.get("status", "unknown")
    if status in TERMINAL_JOB_STATUSES:
        return {
            "job_id": job_id,
            "status": status,
            "message": job.get("message", ""),
            "cancel_requested": False,
        }

    if status == "queued":
        job = _mark_job_canceled(job_id, "Canceled before processing started")
        return {
            "job_id": job_id,
            "status": "canceled",
            "message": job.get("message", "Canceled before processing started"),
            "cancel_requested": False,
        }

    set_job(
        job_id,
        cancel_requested=True,
        cancel_requested_at=time.time(),
        message="Stopping...",
    )
    return {
        "job_id": job_id,
        "status": job.get("status", "running"),
        "message": "Stopping...",
        "cancel_requested": True,
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if _maybe_cleanup_job(job_id, job):
        raise HTTPException(status_code=404, detail="Job expired")

    payload = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
        "cancel_requested": bool(job.get("cancel_requested")),
    }
    if job.get("upload_progress") is not None:
        payload["upload_progress"] = job.get("upload_progress")
    if job.get("tool_used"):
        payload["tool_used"] = job.get("tool_used")
    if job.get("response_type"):
        payload["response_type"] = job.get("response_type")
    if job.get("status") in {"running", "uploading"} and job.get("partial_results"):
        payload["partial_results"] = job.get("partial_results")
        payload["partial_results_total"] = job.get("partial_results_total", 0)
    if job.get("timings"):
        payload["timings"] = job.get("timings")

    if job.get("status") == "completed":
        result_data = job.get("result", {})
        payload["result"] = result_data
    if job.get("status") == "failed":
        payload["error"] = job.get("error", "Unknown error")

    return payload


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload_enabled = os.getenv("API_RELOAD", "0").lower() in {"1", "true", "yes"}
    uvicorn.run(app, host=host, port=port, reload=reload_enabled)
