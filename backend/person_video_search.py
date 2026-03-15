import base64
import json
import logging
import os
import re
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

import config
from preprocess import resolve_preprocessed_dir
from video_reader import VideoReader

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None

config.configure_logging()
logger = logging.getLogger(__name__)

# Defaults
DEFAULT_VIDEO_PATH = "D2C.AVI"
DEFAULT_REFERENCE_IMAGE_PATH = "face.png"
DEFAULT_SCAN_FPS = 2      
DEFAULT_MIN_CONFIDENCE = 0.3
DEFAULT_OUTPUT_DIR = "evidence"
DEFAULT_MAX_MATCHES = 3
DEFAULT_COOLDOWN_SEC = 0.5
DEFAULT_BATCH_SIZE = 10         # Frames per Gemini API call (bigger = fewer calls)
DEFAULT_DUPLICATE_THRESHOLD = 5.0  # Frame-diff threshold to skip near-dupes
DEFAULT_MAX_SIDE = 0            # 0 = no downscaling (keep original resolution)
DEFAULT_PREPROCESS_STRIDE = 1   # Only load every Nth preprocessed frame

# ── InsightFace (RetinaFace + ArcFace) ──────────────────────────────────
DEFAULT_ARCFACE_THRESHOLD = 0.2   # Loose — catch all possible matches → send to Gemini
DEFAULT_ARCFACE_STRONG = 0.55      # Strong local match — auto-accept, skip Gemini

_INSIGHT_APP = None

def _get_face_analyzer():
    """Lazily init InsightFace (RetinaFace detector + ArcFace embedder)."""
    global _INSIGHT_APP
    if _INSIGHT_APP is None:
        try:
            import insightface
            _INSIGHT_APP = insightface.app.FaceAnalysis(
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                allowed_modules=['detection', 'recognition'],  # Skip age/gender for speed
            )
            _INSIGHT_APP.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace loaded (RetinaFace + ArcFace)")
        except Exception as exc:
            logger.warning("Failed to load InsightFace: %s", exc)
            _INSIGHT_APP = None
    return _INSIGHT_APP


def _cosine_similarity(a, b):
    """Cosine similarity between two embedding vectors."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _get_face_embedding(frame):
    """Detect the largest face in frame and return its 512-dim ArcFace embedding.
    Returns (embedding, bbox, det_score) or (None, None, 0.0).
    """
    app = _get_face_analyzer()
    if app is None:
        return None, None, 0.0
    try:
        faces = app.get(frame)
        if not faces:
            return None, None, 0.0
        # Pick the largest face (by bbox area)
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best.embedding, best.bbox.tolist(), float(best.det_score)
    except Exception as exc:
        logger.debug("Face embedding extraction failed: %s", exc)
        return None, None, 0.0


def _get_all_face_embeddings(frame):
    """Detect ALL faces in frame. Returns list of (embedding, bbox, det_score)."""
    app = _get_face_analyzer()
    if app is None:
        return []
    try:
        faces = app.get(frame)
        return [(f.embedding, f.bbox.tolist(), float(f.det_score)) for f in faces]
    except Exception:
        return []


def _format_timestamp(seconds):
    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    hours = total_ms // 3600000
    total_ms %= 3600000
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _clamp_confidence(value):
    if value is None:
        return 0.0
    try:
        conf = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        if not text:
            return 0.0
        if text.endswith("%"):
            text = text[:-1].strip()
            try:
                conf = float(text) / 100.0
            except ValueError:
                return 0.0
        else:
            found = re.search(r"[-+]?\d*\.?\d+", text)
            if not found:
                return 0.0
            conf = float(found.group(0))
    if conf > 1.0 and conf <= 100.0:
        conf = conf / 100.0
    return max(0.0, min(1.0, float(conf)))


def _parse_json_text(text):
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    if "```" in raw:
        raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    # Try to repair truncated JSON (e.g. response cut off mid-array)
    if start >= 0:
        fragment = raw[start:]
        # Close any open strings, arrays, objects
        open_braces = fragment.count("{") - fragment.count("}")
        open_brackets = fragment.count("[") - fragment.count("]")
        # If inside a string, close it
        quote_count = fragment.count('"') - fragment.count('\\"')
        if quote_count % 2 == 1:
            fragment += '"'
        # Close brackets BEFORE braces: {"matches": [...]} 
        fragment += "]" * max(0, open_brackets)
        fragment += "}" * max(0, open_braces)
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            pass
        # Last resort: strip to last complete object "}" in original and close
        original = raw[start:]
        last_close = original.rfind("}")
        if last_close > 0:
            trimmed = original[:last_close + 1]
            open_b = trimmed.count("[") - trimmed.count("]")
            open_o = trimmed.count("{") - trimmed.count("}")
            trimmed += "]" * max(0, open_b) + "}" * max(0, open_o)
            try:
                return json.loads(trimmed)
            except json.JSONDecodeError:
                pass
    return None


def _resize_frame(frame, max_side):
    """Resize frame so longest side <= max_side."""
    if frame is None or frame.size == 0 or max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_frame_jpeg(frame, quality=85, max_side=DEFAULT_MAX_SIDE):
    """Resize + JPEG encode a frame, return bytes."""
    frame = _resize_frame(frame, max_side)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buf.tobytes()


def _frame_diff(frame_a, frame_b):
    """Quick mean-absolute-difference between two frames (grayscale)."""
    try:
        g_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        g_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        # Resize to same small size for fast comparison
        g_a = cv2.resize(g_a, (64, 64))
        g_b = cv2.resize(g_b, (64, 64))
        return float(np.mean(np.abs(g_a.astype(float) - g_b.astype(float))))
    except Exception:
        return 999.0  # Treat errors as "different"


def _require_gemini_client():
    """Create and return a Gemini client."""
    if not genai:
        raise RuntimeError("google-genai is not installed. Install: pip install google-genai")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        api_key = getattr(config, "GEMINI_API_KEY", "").strip()
    if not api_key or "YOUR_KEY" in api_key:
        raise RuntimeError("GEMINI_API_KEY is required. Set it in your environment or config.")
    return genai.Client(api_key=api_key)


class PersonVideoSearch:
    """
    Hybrid person search using RetinaFace + ArcFace + Gemini.
    - RetinaFace: accurate face detection in each frame
    - ArcFace: 512-dim face embedding comparison (cosine similarity)
    - Gemini: 'final boss' verification for borderline matches
    Strong ArcFace matches (sim >= 0.55) are auto-accepted.
    Candidates (0.35 <= sim < 0.55) go to Gemini for verification.
    """

    def __init__(self):
        self.gemini_client = None
        self.model_name = getattr(config, "GEMINI_MODEL", "gemini-2.0-flash")
        try:
            self.gemini_client = _require_gemini_client()
            logger.info("PersonVideoSearch initialized with Gemini model: %s", self.model_name)
        except Exception as exc:
            logger.error("Failed to initialize Gemini client: %s", exc)
            raise

    def _load_preprocessed_frames(self, pre_dir, duplicate_threshold, stride=DEFAULT_PREPROCESS_STRIDE):
        """
        Load pre-extracted frames from a preprocessed directory.
        Only loads every `stride`-th frame to avoid sending too many.
        Returns list of {frame, frame_idx, time_sec}.
        """
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
        all_frame_files = sorted(
            name for name in os.listdir(pre_dir)
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS
            and name != "preview.jpg"
        )
        if not all_frame_files:
            return [], 0

        # Sub-sample: only take every stride-th file
        frame_files = all_frame_files[::max(1, stride)]

        # Read meta.json for FPS info
        meta_path = os.path.join(pre_dir, "meta.json")
        source_fps = 30.0
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                source_fps = float(meta.get("source_fps", 30.0))
            except Exception:
                pass

        sampled = []
        prev_frame = None

        for name in tqdm(frame_files, desc="Loading preprocessed frames"):
            path = os.path.join(pre_dir, name)
            frame = cv2.imread(path)
            if frame is None:
                continue

            # Parse frame_idx and timestamp from filename
            # Pattern: frame_000000_idx_0_t_00-00-00.000.jpg
            frame_idx = 0
            time_sec = 0.0
            try:
                parts = name.replace(".jpg", "").replace(".png", "").split("_")
                idx_pos = parts.index("idx") if "idx" in parts else -1
                if idx_pos >= 0 and idx_pos + 1 < len(parts):
                    frame_idx = int(parts[idx_pos + 1])
                    time_sec = frame_idx / source_fps
                t_pos = parts.index("t") if "t" in parts else -1
                if t_pos >= 0 and t_pos + 1 < len(parts):
                    ts_str = parts[t_pos + 1]
                    # Parse HH-MM-SS.mmm
                    ts_parts = ts_str.split("-")
                    if len(ts_parts) == 3:
                        h, m = int(ts_parts[0]), int(ts_parts[1])
                        s_ms = ts_parts[2].split(".")
                        s = int(s_ms[0])
                        ms = int(s_ms[1]) if len(s_ms) > 1 else 0
                        time_sec = h * 3600 + m * 60 + s + ms / 1000.0
            except (ValueError, IndexError):
                pass

            # Skip near-duplicates
            if prev_frame is not None and duplicate_threshold > 0:
                diff = _frame_diff(frame, prev_frame)
                if diff < duplicate_threshold:
                    continue

            sampled.append({
                "frame": frame,
                "frame_idx": frame_idx,
                "time_sec": time_sec,
            })
            prev_frame = frame

        logger.info("Loaded %d frames from %s (stride=%d, %d total files)",
                    len(sampled), pre_dir, stride, len(all_frame_files))
        return sampled, source_fps

    def _sample_frames(self, video_path, scan_fps, duplicate_threshold):
        """
        Extract frames from video at scan_fps, skip near-duplicates.
        Returns list of {frame, frame_idx, time_sec}.
        """
        cap = VideoReader(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"video_open_failed: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = getattr(config, "FPS", 30)
        frame_stride = max(1, int(round(float(fps) / scan_fps)))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        sampled = []
        prev_frame = None
        frame_idx = 0

        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Sampling frames")

        while cap.isOpened():
            if frame_idx % frame_stride == 0:
                # Capture timestamp BEFORE read() — OpenCV advances position after read
                try:
                    time_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                except Exception:
                    time_msec = 0.0

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                # Skip near-duplicate frames
                if prev_frame is not None:
                    diff = _frame_diff(frame, prev_frame)
                    if diff < duplicate_threshold:
                        frame_idx += 1
                        pbar.update(1)
                        continue

                if time_msec > 0:
                    time_sec = time_msec / 1000.0
                else:
                    time_sec = frame_idx / float(fps)

                sampled.append({
                    "frame": frame.copy(),
                    "frame_idx": frame_idx,
                    "time_sec": time_sec,
                })
                prev_frame = frame.copy()

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        logger.info("Sampled %d frames from %d total (stride=%d, dup_thresh=%.1f)",
                     len(sampled), total_frames, frame_stride, duplicate_threshold)
        return sampled, fps

    def _gemini_compare_batch(self, reference_bytes, frame_batch, query_text=""):
        """
        Send reference image + a batch of numbered video frames to Gemini.
        Ask which frames contain the same person.
        Returns list of {frame_number, confidence, reason} for matches.
        """
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

        contents = self._build_batch_contents(reference_bytes, frame_batch, query_text)

        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            raw_text = response.text or ""
            parsed = _parse_json_text(raw_text)
            if parsed and isinstance(parsed.get("matches"), list):
                return parsed["matches"]
            logger.warning("Gemini returned unparseable response: %s", raw_text[:200])
            return []
        except Exception as exc:
            logger.warning("Gemini API error: %s", exc)
            return []

    async def _gemini_compare_batch_async(self, reference_bytes, frame_batch, query_text=""):
        """Async version — uses client.aio for non-blocking Gemini calls."""
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

        contents = self._build_batch_contents(reference_bytes, frame_batch, query_text)

        try:
            response = await self.gemini_client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            raw_text = response.text or ""
            parsed = _parse_json_text(raw_text)
            if parsed and isinstance(parsed.get("matches"), list):
                return parsed["matches"]
            logger.warning("Gemini returned unparseable response: %s", raw_text[:200])
            return []
        except Exception as exc:
            logger.warning("Gemini async API error: %s", exc)
            return []

    def _build_batch_contents(self, reference_bytes, frame_batch, query_text=""):
        """Build the Gemini content payload for a batch of frames."""
        contents = []

        # Add reference image
        ref_part = genai_types.Part.from_bytes(data=reference_bytes, mime_type="image/jpeg")
        contents.append(ref_part)
        contents.append("Above is the REFERENCE person image. Find this EXACT person in the numbered frames below.\n")

        # Add numbered frames
        for i, item in enumerate(frame_batch):
            frame_bytes = _encode_frame_jpeg(item["frame"])
            if frame_bytes is None:
                continue
            frame_part = genai_types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg")
            contents.append(f"Frame {i + 1} (timestamp: {_format_timestamp(item['time_sec'])}):")
            contents.append(frame_part)

        # Build prompt — FACE ONLY, short answers
        prompt = (
            "\nIDENTIFY which frames show the SAME PERSON as the reference by comparing FACES ONLY. "
            "Compare facial structure: face shape, eyes, nose, mouth, jawline, eyebrows, skin tone. "
            "IGNORE clothing, hair, body shape, posture, and accessories completely. "
            "Only match if a face is clearly visible and matches the reference face. "
        )
        if query_text:
            prompt += f"Context: {query_text}. "

        prompt += (
            "\nReturn ONLY the TOP 2 matches with HIGHEST confidence.\n"
            "Return ONLY valid JSON, no markdown fences:\n"
            '{"matches": [{"frame_number": 1, "confidence": 0.85, "reason": "same face"}]}\n'
            'Keep "reason" to MAX 5 words. If no match: {"matches": []}\n'
            "Only report matches with confidence >= 0.6."
        )
        contents.append(prompt)
        return contents

    async def _run_gemini_batches_async(self, reference_bytes, batches, query_text):
        """Fire all Gemini batches concurrently as async tasks, resolve with gather."""
        tasks = [
            self._gemini_compare_batch_async(reference_bytes, batch, query_text)
            for batch in batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Replace exceptions with empty lists
        return [
            r if isinstance(r, list) else [] for r in results
        ]



    def search(
        self,
        video_path,
        reference_image_path,
        query_text="",
        scan_fps=DEFAULT_SCAN_FPS,
        min_confidence=DEFAULT_MIN_CONFIDENCE,
        max_matches=DEFAULT_MAX_MATCHES,
        cooldown_sec=DEFAULT_COOLDOWN_SEC,
        output_dir=DEFAULT_OUTPUT_DIR,
        show_progress=True,
        batch_size=DEFAULT_BATCH_SIZE,
        duplicate_threshold=DEFAULT_DUPLICATE_THRESHOLD,
        preprocessed_dir=None,
        # Accept but ignore legacy params for backward compat
        use_face_matching=None,
        face_similarity_threshold=None,
        face_min_size=None,
        gemini_verify=None,
        gemini_max_candidates=None,
        candidate_min_gap_sec=None,
    ):
        """
        Search for a person in a video using Gemini.
        
        Primary: Upload video directly to Gemini (single fast API call)
        Fallback: Batch frame-by-frame comparison
        """
        video_path = str(video_path)
        reference_image_path = str(reference_image_path)
        scan_fps = max(0.2, float(scan_fps))
        min_confidence = max(0.0, min(1.0, float(min_confidence)))
        max_matches = max(1, int(max_matches))
        cooldown_sec = max(0.0, float(cooldown_sec))
        batch_size = max(1, min(10, int(batch_size)))

        # Load reference image
        reference = cv2.imread(reference_image_path)
        if reference is None or reference.size == 0:
            raise RuntimeError(f"reference_image_unreadable: {reference_image_path}")

        reference_bytes = _encode_frame_jpeg(reference, quality=90, max_side=DEFAULT_MAX_SIDE)
        if reference_bytes is None:
            raise RuntimeError("Failed to encode reference image")

        # Setup output directories
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"person_search_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir = run_dir / "frames"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # Debug directory setup
        debug_dir = Path("person_search_debug") / run_tag
        debug_dirs = {
            "1_reference": debug_dir / "1_reference",
            "2_sampled": debug_dir / "2_sampled_frames",
            "3_arcface": debug_dir / "3_arcface_matches",
            "3b_candidates": debug_dir / "3b_gemini_candidates",
            "4_gemini": debug_dir / "4_gemini_io",
            "5_matches": debug_dir / "5_matches",
        }
        for d in debug_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Debug output → %s", debug_dir)

        start_time = time.time()

        # Extract reference face embedding (ArcFace)
        logger.info("Extracting reference face embedding...")
        ref_embedding, ref_bbox, ref_det_score = _get_face_embedding(reference)
        if ref_embedding is None:
            logger.warning("No face detected in reference image — will rely on Gemini only")

        # Debug: Save reference image + embedding info
        try:
            cv2.imwrite(str(debug_dirs["1_reference"] / "reference.jpg"), reference)
            ref_info = {"path": reference_image_path, "video": video_path,
                        "query": query_text, "scan_fps": scan_fps,
                        "min_confidence": min_confidence, "max_matches": max_matches,
                        "ref_face_detected": ref_embedding is not None,
                        "ref_det_score": round(ref_det_score, 3),
                        "ref_bbox": ref_bbox,
                        "arcface_threshold": DEFAULT_ARCFACE_THRESHOLD,
                        "arcface_strong": DEFAULT_ARCFACE_STRONG}
            with open(debug_dirs["1_reference"] / "info.json", "w") as f:
                json.dump(ref_info, f, indent=2)
        except Exception:
            pass

        # Step 1: Load/sample frames
        if preprocessed_dir and os.path.isdir(preprocessed_dir):
            logger.info("Step 1: Loading preprocessed frames from %s", preprocessed_dir)
            sampled_frames, fps = self._load_preprocessed_frames(
                preprocessed_dir, duplicate_threshold
            )
        else:
            logger.info("Step 1: Sampling frames from video at %.1f fps...", scan_fps)
            sampled_frames, fps = self._sample_frames(
                video_path, scan_fps, duplicate_threshold
            )

        if not sampled_frames:
            logger.warning("No frames sampled from video")
            return self._build_report(
                video_path, reference_image_path, query_text,
                scan_fps, min_confidence, max_matches,
                [], run_dir, start_time
            )

        logger.info("Sampled %d frames total", len(sampled_frames))

        # Debug: Save all sampled frames
        for i, sf in enumerate(sampled_frames):
            try:
                ts = _format_timestamp(sf['time_sec']).replace(':', '-')
                cv2.imwrite(str(debug_dirs["2_sampled"] / f"frame_{i:04d}_t{ts}_idx{sf['frame_idx']}.jpg"),
                           sf["frame"])
            except Exception:
                pass
        logger.info("Debug: saved %d sampled frames", len(sampled_frames))

        # ─── Step 1.5: ArcFace embedding comparison ─────────────────────
        arcface_start = time.time()
        strong_matches = []   # sim >= STRONG → auto-accept
        candidates = []       # THRESHOLD <= sim < STRONG → send to Gemini
        arcface_details = []  # Debug log

        if ref_embedding is not None:
            logger.info("Step 1.5: ArcFace matching (%d frames, threshold=%.2f, strong=%.2f)...",
                        len(sampled_frames), DEFAULT_ARCFACE_THRESHOLD, DEFAULT_ARCFACE_STRONG)

            # Process frames in parallel (ONNX releases GIL)
            def _process_one_frame(sf_i):
                sf, i = sf_i
                frame = sf["frame"]
                all_faces = _get_all_face_embeddings(frame)
                best_sim = 0.0
                best_bbox = None
                best_det = 0.0
                for emb, bbox, det_score in all_faces:
                    sim = _cosine_similarity(ref_embedding, emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_bbox = bbox
                        best_det = det_score
                return i, sf, all_faces, best_sim, best_bbox, best_det

            n_arcface_workers = min(4, max(1, len(sampled_frames)))
            with ThreadPoolExecutor(max_workers=n_arcface_workers) as pool:
                futures = pool.map(_process_one_frame,
                                   [(sf, i) for i, sf in enumerate(sampled_frames)])

                for i, sf, all_faces, best_sim, best_bbox, best_det in futures:
                    detail = {
                        "frame_idx": sf["frame_idx"],
                        "time_sec": sf["time_sec"],
                        "n_faces": len(all_faces),
                        "best_similarity": round(best_sim, 4),
                        "best_det_score": round(best_det, 3),
                        "category": "no_face"
                    }

                    if best_sim >= DEFAULT_ARCFACE_STRONG:
                        detail["category"] = "strong_match"
                        sf["arcface_sim"] = best_sim
                        sf["face_bbox"] = best_bbox
                        strong_matches.append(sf)
                    elif best_sim >= DEFAULT_ARCFACE_THRESHOLD:
                        detail["category"] = "candidate"
                        sf["arcface_sim"] = best_sim
                        sf["face_bbox"] = best_bbox
                        candidates.append(sf)
                    elif len(all_faces) > 0:
                        detail["category"] = "below_threshold"

                    arcface_details.append(detail)

            arcface_time = time.time() - arcface_start
            logger.info("ArcFace: %d strong + %d candidates + %d rejected (%.1fs)",
                        len(strong_matches), len(candidates),
                        len(sampled_frames) - len(strong_matches) - len(candidates),
                        arcface_time)
        else:
            # No reference embedding — fall back to sending ALL frames to Gemini
            logger.warning("No reference face embedding — sending all frames to Gemini")
            candidates = sampled_frames
            arcface_time = 0.0

        # Debug: Save ArcFace results
        try:
            with open(debug_dirs["3_arcface"] / "arcface_results.json", "w") as f:
                json.dump({
                    "total_sampled": len(sampled_frames),
                    "strong_matches": len(strong_matches),
                    "candidates_for_gemini": len(candidates),
                    "arcface_time_sec": round(arcface_time, 3),
                    "threshold": DEFAULT_ARCFACE_THRESHOLD,
                    "strong_threshold": DEFAULT_ARCFACE_STRONG,
                    "details": arcface_details
                }, f, indent=2)
        except Exception:
            pass

        # ─── Step 2: Gemini verification on candidates only ──────────────
        #   Strong matches are auto-accepted; candidates go to Gemini as "final boss"
        gemini_verified = []

        if candidates:
            logger.info("Step 2: Sending %d candidate frames to Gemini for verification...", len(candidates))
            # Debug: Save candidates being sent to Gemini
            for i, c in enumerate(candidates):
                try:
                    ts = _format_timestamp(c['time_sec']).replace(':', '-')
                    cv2.imwrite(str(debug_dirs["3b_candidates"] / f"candidate_{i:04d}_sim{c.get('arcface_sim', 0):.3f}_t{ts}.jpg"),
                               c["frame"])
                except Exception:
                    pass

            # Send candidate frames directly to Gemini as images (batch comparison)
            batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
            n_workers = min(len(batches), 4)
            logger.info("Sending %d candidates in %d batches (%d workers) to Gemini...",
                        len(candidates), len(batches), n_workers)

            # Debug: Save prompt info
            try:
                with open(debug_dirs["4_gemini"] / "request_info.json", "w") as f:
                    json.dump({"model": self.model_name, "n_candidates": len(candidates),
                               "n_batches": len(batches), "batch_size": batch_size,
                               "query": query_text, "max_matches": max_matches}, f, indent=2)
            except Exception:
                pass

            # Fire all batches concurrently (async)
            logger.info("Firing %d Gemini batches concurrently (async)...", len(batches))
            batch_results = asyncio.run(
                self._run_gemini_batches_async(reference_bytes, batches, query_text)
            )

            # Map batch results back to candidate frames
            for batch_idx, matches in enumerate(batch_results):
                if not matches:
                    continue
                batch = batches[batch_idx]
                for bm in matches:
                    frame_num = bm.get("frame_number", 0)
                    idx = frame_num - 1
                    if 0 <= idx < len(batch):
                        fi = batch[idx]
                        gemini_verified.append({
                            "timestamp_sec": fi["time_sec"],
                            "confidence": bm.get("confidence", 0),
                            "reason": bm.get("reason", ""),
                            "method": "arcface+gemini",
                            "arcface_sim": fi.get("arcface_sim", 0),
                            "frame_idx": fi.get("frame_idx", 0),
                            "face_bbox": fi.get("face_bbox"),
                            "_frame": fi.get("frame"),  # actual frame for evidence
                        })

            logger.info("Gemini verified %d/%d candidates", len(gemini_verified), len(candidates))

            # Debug: Save Gemini results
            try:
                with open(debug_dirs["4_gemini"] / "gemini_results.json", "w") as f:
                    json.dump({"n_candidates": len(candidates), "n_verified": len(gemini_verified),
                               "verified": gemini_verified}, f, indent=2, default=str)
            except Exception:
                pass
        else:
            logger.info("Step 2: No candidates for Gemini (all strong or all rejected)")

        # ─── Step 3: Combine strong matches + Gemini-verified ────────────
        # Build combined results
        all_matches = []

        # Add strong matches (auto-accepted by ArcFace)
        for sf in strong_matches:
            all_matches.append({
                "timestamp_sec": sf["time_sec"],
                "confidence": round(sf["arcface_sim"], 4),
                "reason": "arcface strong match",
                "method": "arcface_auto",
                "frame_idx": sf["frame_idx"],
                "face_bbox": sf.get("face_bbox"),
                "_frame": sf["frame"],  # Keep actual frame for evidence
            })

        # Add Gemini-verified matches (frame + bbox already stored)
        for vm in gemini_verified:
            all_matches.append(vm)

        # Sort by timestamp
        all_matches.sort(key=lambda m: m.get("timestamp_sec", 0))

        # Debug: Save combined matches (without frame data)
        try:
            debug_matches = [{k: v for k, v in m.items() if k != "_frame"} for m in all_matches]
            with open(debug_dirs["5_matches"] / "all_matches.json", "w") as f:
                json.dump({
                    "n_strong": len(strong_matches),
                    "n_gemini_verified": len(gemini_verified),
                    "total": len(all_matches),
                    "matches": debug_matches
                }, f, indent=2, default=str)
        except Exception:
            pass

        logger.info("Combined: %d strong + %d Gemini-verified = %d total matches",
                    len(strong_matches), len(gemini_verified), len(all_matches))

        if not all_matches:
            return self._build_report(
                video_path, reference_image_path, query_text,
                scan_fps, min_confidence, max_matches,
                [], run_dir, start_time
            )

        # ─── Save evidence frames directly (with video fallback) ──────────
        # Sort by confidence descending — keep the best matches
        all_matches.sort(key=lambda m: m.get("confidence", 0), reverse=True)

        results = []
        last_match_sec = -999999.0
        cap = None  # Lazy-open video only if needed

        for match in all_matches:
            if len(results) >= max_matches:
                break

            time_sec = float(match.get("timestamp_sec", 0))
            confidence = _clamp_confidence(match.get("confidence", 0))
            reason = match.get("reason", "")
            frame = match.get("_frame")
            frame_idx = match.get("frame_idx", int(time_sec * fps))

            if confidence < min_confidence:
                continue
            if (time_sec - last_match_sec) < cooldown_sec:
                continue

            # Fallback: read frame from video if not in memory
            if frame is None:
                try:
                    if cap is None:
                        cap = VideoReader(video_path)
                    # Use frame-based seeking (exact) instead of time-based (inaccurate)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        frame = None
                    else:
                        logger.info("Fallback: read frame %d from video for evidence", frame_idx)
                except Exception:
                    frame = None

            # Save evidence frame with bounding box drawn on it
            frame_path = None
            if frame is not None:
                evidence_frame = frame.copy()
                face_bbox = match.get("face_bbox")

                # If no bbox stored, re-detect face on this frame
                if not face_bbox or len(face_bbox) != 4:
                    logger.info("No stored bbox at %.1fs — running face detection on evidence frame", time_sec)
                    try:
                        _, detected_bbox, _ = _get_face_embedding(evidence_frame)
                        if detected_bbox:
                            face_bbox = detected_bbox
                            logger.info("Re-detected face bbox: %s", face_bbox)
                    except Exception as exc:
                        logger.warning("Face re-detection failed at %.1fs: %s", time_sec, exc)

                # Draw face bounding box
                if face_bbox and len(face_bbox) == 4:
                    try:
                        x1, y1, x2, y2 = int(face_bbox[0]), int(face_bbox[1]), int(face_bbox[2]), int(face_bbox[3])
                        # Green bbox — thick for CCTV visibility
                        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        # Label above bbox
                        label = f"{confidence:.0%} {match.get('method', '')}"
                        label_y = max(y1 - 10, 20)
                        cv2.putText(evidence_frame, label, (x1, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        logger.info("Drew bbox [%d,%d,%d,%d] on evidence at %.1fs",
                                    x1, y1, x2, y2, time_sec)
                    except Exception as exc:
                        logger.warning("Failed to draw bbox at %.1fs: %s (bbox=%s)", time_sec, exc, face_bbox)
                else:
                    logger.warning("No face found for bbox at %.1fs (method=%s)", time_sec, match.get("method"))

                # Save the frame (with or without bbox)
                try:
                    match_num = len(results) + 1
                    stamp = _format_timestamp(time_sec).replace(":", "-")
                    frame_name = f"match_{match_num:03d}_frame_{frame_idx}_t_{stamp}.jpg"
                    fpath = evidence_dir / frame_name
                    cv2.imwrite(str(fpath), evidence_frame)
                    frame_path = str(fpath)
                    logger.info("Saved evidence frame: %s (bbox=%s)", frame_name, "YES" if face_bbox else "NO")
                except Exception as exc:
                    logger.warning("Failed to save evidence frame at %.1fs: %s", time_sec, exc)

                # Generate base64 thumbnail for frontend
                try:
                    _, thumb_buf = cv2.imencode(".jpg", evidence_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    thumb_b64 = base64.b64encode(thumb_buf.tobytes()).decode("utf-8")
                except Exception:
                    thumb_b64 = None
            else:
                logger.warning("No frame data for match at %.1fs — cannot save evidence", time_sec)
                thumb_b64 = None

            result = {
                "type": "person_match",
                "time_sec": round(time_sec, 3),
                "timestamp": _format_timestamp(time_sec),
                "frame": frame_idx,
                "confidence": round(confidence, 4),
                "method": match.get("method", "hybrid"),
                "reason": str(reason).strip(),
            }
            if frame_path:
                result["evidence_frame_path"] = frame_path
            if thumb_b64:
                result["thumbnail"] = f"data:image/jpeg;base64,{thumb_b64}"

            results.append(result)
            last_match_sec = time_sec
            logger.info("Person found at %s (frame=%d, confidence=%.3f, reason=%s)",
                        result["timestamp"], frame_idx, confidence, reason[:60])

        if cap:
            cap.release()

        return self._build_report(
            video_path, reference_image_path, query_text,
            scan_fps, min_confidence, max_matches,
            results, run_dir, start_time
        )

    def _process_video_matches(self, video_path, video_matches, reference_image_path,
                               query_text, scan_fps, min_confidence, max_matches,
                               cooldown_sec, evidence_dir, run_dir, start_time):
        """Process matches from Gemini video search — extract evidence frames."""
        cap = VideoReader(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video for evidence extraction")
            cap = None

        fps = cap.get(cv2.CAP_PROP_FPS) if cap else 30.0
        if not fps or fps <= 0:
            fps = 30.0

        results = []
        last_match_sec = -999999.0

        for match in video_matches:
            if len(results) >= max_matches:
                break

            time_sec = float(match.get("timestamp_sec", 0))
            confidence = _clamp_confidence(match.get("confidence", 0))
            reason = match.get("reason", "")

            if confidence < min_confidence:
                continue
            if (time_sec - last_match_sec) < cooldown_sec:
                continue

            frame_idx = int(time_sec * fps)

            # Extract the frame at this timestamp for evidence
            frame_path = None
            if cap:
                try:
                    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Draw face bounding box on evidence frame
                        face_bbox = match.get("face_bbox")
                        if face_bbox is None:
                            # Re-detect face for bbox if not stored
                            _, face_bbox, _ = _get_face_embedding(frame)
                        if face_bbox:
                            x1, y1, x2, y2 = [int(v) for v in face_bbox]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{confidence:.0%} {reason[:20]}"
                            cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        match_num = len(results) + 1
                        stamp = _format_timestamp(time_sec).replace(":", "-")
                        frame_name = f"match_{match_num:03d}_frame_{frame_idx}_t_{stamp}.jpg"
                        fpath = evidence_dir / frame_name
                        cv2.imwrite(str(fpath), frame)
                        frame_path = str(fpath)
                except Exception as exc:
                    logger.warning("Failed to extract evidence frame at %.1fs: %s", time_sec, exc)

            result = {
                "type": "person_match",
                "time_sec": round(time_sec, 3),
                "timestamp": _format_timestamp(time_sec),
                "frame": frame_idx,
                "confidence": round(confidence, 4),
                "method": "gemini_video",
                "reason": str(reason).strip(),
            }
            if frame_path:
                result["evidence_frame_path"] = frame_path

            results.append(result)
            last_match_sec = time_sec
            logger.info("Person found at %s (confidence=%.3f, reason=%s)",
                        result["timestamp"], confidence, reason[:60])

        if cap:
            cap.release()

        return self._build_report(
            video_path, reference_image_path, query_text,
            scan_fps, min_confidence, max_matches,
            results, run_dir, start_time
        )

    def _build_report(self, video_path, reference_image_path, query_text,
                      scan_fps, min_confidence, max_matches,
                      results, run_dir, start_time):
        """Build and save the final report."""
        report = {
            "query": query_text,
            "video": video_path,
            "reference_image": reference_image_path,
            "tool_used": "FACE_SEARCH",
            "events_found": len(results),
            "scan_fps": scan_fps,
            "min_confidence": min_confidence,
            "max_matches": max_matches,
            "method": "hybrid_arcface_gemini",
            "gemini_model": self.model_name,
            "results": results,
            "output_dir": str(run_dir),
            "elapsed_sec": round(time.time() - start_time, 3),
        }

        report_path = run_dir / "report.json"
        try:
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            report["report_path"] = str(report_path)
        except Exception as exc:
            logger.warning("Failed to save report: %s", exc)

        logger.info("Person search complete: matches=%d elapsed=%.1fs report=%s",
                     len(results), report["elapsed_sec"], report_path)
        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Find a person from reference image in video using Gemini."
    )
    parser.add_argument("video_path", nargs="?", default=DEFAULT_VIDEO_PATH,
                        help="Path to input video")
    parser.add_argument("reference_image_path", nargs="?", default=DEFAULT_REFERENCE_IMAGE_PATH,
                        help="Path to reference person image")
    parser.add_argument("--query", default="", help="Optional context (e.g. 'person in red jacket')")
    parser.add_argument("--scan-fps", type=float, default=DEFAULT_SCAN_FPS,
                        help="Frames per second to sample")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE,
                        help="Minimum confidence to accept a match")
    parser.add_argument("--max-matches", type=int, default=DEFAULT_MAX_MATCHES,
                        help="Maximum matches to return")
    parser.add_argument("--cooldown-sec", type=float, default=DEFAULT_COOLDOWN_SEC,
                        help="Minimum gap between matches in seconds")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Frames per Gemini API call")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output folder for evidence")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    args = parser.parse_args()

    engine = PersonVideoSearch()
    report = engine.search(
        video_path=args.video_path,
        reference_image_path=args.reference_image_path,
        query_text=args.query,
        scan_fps=args.scan_fps,
        min_confidence=args.min_confidence,
        max_matches=args.max_matches,
        cooldown_sec=args.cooldown_sec,
        output_dir=args.output_dir,
        show_progress=(not args.no_progress),
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
