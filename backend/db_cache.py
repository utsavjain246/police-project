import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import time

from fastapi import HTTPException

import shutil

logger = logging.getLogger(__name__)

VIDEO_ID_MAX_LEN = 128
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


class ForensicDB:
    def __init__(self, db_path, video_store_dir):
        self.db_path = db_path
        self.video_store_dir = video_store_dir

    def init_db(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        os.makedirs(self.video_store_dir, exist_ok=True)
        db_exists = os.path.exists(self.db_path)
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            if db_exists:
                logger.info("Using existing forensic DB (ensuring schema): %s", self.db_path)
            else:
                logger.info("Creating new forensic DB: %s", self.db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_ref TEXT,
                    storage_path TEXT,
                    size_bytes INTEGER,
                    checksum_sha256 TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    video_id TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    query_norm TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (video_id, query_hash),
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS domain_aggregate (
                    video_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    aggregate_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (video_id, tool_name),
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    def ensure_ready(self):
        if not os.path.exists(self.db_path):
            self.init_db()
        if not os.path.isdir(self.video_store_dir):
            os.makedirs(self.video_store_dir, exist_ok=True)

    @staticmethod
    def normalize_query(query):
        return " ".join((query or "").strip().lower().split())

    @staticmethod
    def query_hash(query_norm):
        return hashlib.sha256(query_norm.encode("utf-8")).hexdigest()

    def validate_video_id(self, raw_video_id):
        value = (raw_video_id or "").strip()
        if not value:
            raise HTTPException(status_code=400, detail="video_id cannot be empty")
        if len(value) > VIDEO_ID_MAX_LEN:
            raise HTTPException(status_code=400, detail="video_id too long")
        if not VIDEO_ID_PATTERN.fullmatch(value):
            raise HTTPException(
                status_code=400,
                detail="video_id contains invalid characters",
            )
        return value

    @staticmethod
    def sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as file_obj:
            while True:
                chunk = file_obj.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def safe_video_store_name(video_id, suffix):
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id)
        return f"{sanitized}{suffix or '.mp4'}"

    def persist_video_copy(self, video_id, source_path, suffix):
        os.makedirs(self.video_store_dir, exist_ok=True)
        filename = self.safe_video_store_name(video_id, suffix)
        destination = os.path.join(self.video_store_dir, filename)
        return destination if _copy2(source_path, destination) else destination

    def stage_video_copy(self, source_path, suffix):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as tmp_file:
            tmp_path = tmp_file.name
        try:
            _copy2(source_path, tmp_path)
        except Exception as exc:
            _cleanup_file(tmp_path)
            logger.exception("Failed to stage video copy: %s", exc)
            raise HTTPException(status_code=500, detail=f"Failed to stage video: {exc}")
        return tmp_path

    def get_video(self, video_id):
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT video_id, source_type, source_ref, storage_path, size_bytes, checksum_sha256,
                       created_at, updated_at
                FROM videos
                WHERE video_id = ?
                """,
                (video_id,),
            ).fetchone()
            return dict(row) if row else None

    def upsert_video(self, video_id, source_type, source_ref, storage_path, size_bytes, checksum_sha256):
        now = time.time()
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            existing = conn.execute(
                "SELECT checksum_sha256 FROM videos WHERE video_id = ?",
                (video_id,),
            ).fetchone()
            if existing and existing["checksum_sha256"] and checksum_sha256:
                if existing["checksum_sha256"] != checksum_sha256:
                    conn.execute("DELETE FROM analysis_cache WHERE video_id = ?", (video_id,))

            conn.execute(
                """
                INSERT INTO videos (
                    video_id, source_type, source_ref, storage_path, size_bytes, checksum_sha256, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    source_type = excluded.source_type,
                    source_ref = excluded.source_ref,
                    storage_path = excluded.storage_path,
                    size_bytes = excluded.size_bytes,
                    checksum_sha256 = excluded.checksum_sha256,
                    updated_at = excluded.updated_at
                """,
                (
                    video_id,
                    source_type,
                    source_ref,
                    storage_path,
                    size_bytes,
                    checksum_sha256,
                    now,
                    now,
                ),
            )
            conn.commit()

    def get_cached_report(self, video_id, query_norm):
        query_digest = self.query_hash(query_norm)
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT report_json
                FROM analysis_cache
                WHERE video_id = ? AND query_hash = ?
                """,
                (video_id, query_digest),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """
                UPDATE analysis_cache
                SET hit_count = hit_count + 1, updated_at = ?
                WHERE video_id = ? AND query_hash = ?
                """,
                (time.time(), video_id, query_digest),
            )
            conn.commit()
            try:
                return json.loads(row["report_json"])
            except Exception:
                logger.warning("Cached report decode failed for video_id=%s", video_id)
                return None

    def cache_report(self, video_id, query_norm, query_text, report):
        if not isinstance(report, dict):
            return
        query_digest = self.query_hash(query_norm)
        now = time.time()
        payload = json.dumps(report, default=lambda o: int(o) if hasattr(o, 'item') else str(o))
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute(
                """
                INSERT INTO analysis_cache (
                    video_id, query_hash, query_norm, query_text, report_json, created_at, updated_at, hit_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(video_id, query_hash) DO UPDATE SET
                    query_norm = excluded.query_norm,
                    query_text = excluded.query_text,
                    report_json = excluded.report_json,
                    updated_at = excluded.updated_at
                """,
                (video_id, query_digest, query_norm, query_text, payload, now, now),
            )
            conn.commit()

    def _thumbnail_dir(self, video_id, tool_name):
        """Return the directory for storing thumbnails for a domain aggregate."""
        base = os.path.join(os.path.dirname(self.db_path), "cache", "thumbnails", video_id, tool_name)
        return base

    def get_domain_aggregate(self, video_id, tool_name):
        """Return cached aggregate for (video_id, tool_name) or None.
        Thumbnails are stored as file paths — caller must lazy-load as needed.
        """
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT aggregate_json FROM domain_aggregate WHERE video_id = ? AND tool_name = ?",
                (video_id, tool_name),
            ).fetchone()
            if not row:
                return None
            try:
                return json.loads(row["aggregate_json"])
            except Exception:
                logger.warning("Domain aggregate decode failed for video_id=%s tool=%s", video_id, tool_name)
                return None

    def cache_domain_aggregate(self, video_id, tool_name, aggregate_data):
        """Store domain aggregate. Strips base64 thumbnails to disk files, stores paths instead."""
        if not isinstance(aggregate_data, (dict, list)):
            return

        # Strip thumbnails to files
        thumb_dir = self._thumbnail_dir(video_id, tool_name)
        events = aggregate_data.get("events", []) if isinstance(aggregate_data, dict) else aggregate_data
        stripped_events = self._strip_thumbnails_to_disk(events, thumb_dir)

        # Build lightweight aggregate (no base64)
        if isinstance(aggregate_data, dict):
            light_aggregate = {**aggregate_data, "events": stripped_events}
        else:
            light_aggregate = stripped_events

        now = time.time()
        payload = json.dumps(light_aggregate, default=lambda o: int(o) if hasattr(o, 'item') else str(o))
        with sqlite3.connect(self.db_path, timeout=30) as conn:
            conn.execute(
                """
                INSERT INTO domain_aggregate (video_id, tool_name, aggregate_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(video_id, tool_name) DO UPDATE SET
                    aggregate_json = excluded.aggregate_json,
                    created_at = excluded.created_at
                """,
                (video_id, tool_name, payload, now),
            )
            conn.commit()
        logger.info("Cached domain aggregate: video_id=%s tool=%s (%d bytes, %d thumbnails saved)",
                     video_id, tool_name, len(payload), len([e for e in stripped_events if e.get("thumbnail_path")]))

    def _strip_thumbnails_to_disk(self, events, thumb_dir):
        """Extract base64 thumbnails from events, save as JPEG files, replace with file paths."""
        import base64 as b64
        stripped = []
        saved_count = 0
        for idx, event in enumerate(events):
            evt = dict(event)  # shallow copy
            thumbnail = evt.get("thumbnail")
            if thumbnail and isinstance(thumbnail, str) and len(thumbnail) > 200:
                # Looks like base64 image data
                try:
                    os.makedirs(thumb_dir, exist_ok=True)
                    # Strip data URI prefix if present
                    raw = thumbnail
                    if raw.startswith("data:"):
                        raw = raw.split(",", 1)[1] if "," in raw else raw
                    img_bytes = b64.b64decode(raw)
                    fname = f"evt_{idx}.jpg"
                    fpath = os.path.join(thumb_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(img_bytes)
                    evt.pop("thumbnail", None)
                    evt["thumbnail_path"] = fpath
                    saved_count += 1
                except Exception as exc:
                    logger.warning("Failed to save thumbnail for event %d: %s", idx, exc)
            stripped.append(evt)
        if saved_count:
            logger.debug("Saved %d thumbnails to %s", saved_count, thumb_dir)
        return stripped

    @staticmethod
    def load_thumbnail_b64(event):
        """Lazy-load a single event's thumbnail. Returns the URL path instead of decoding to base64 to save bandwidth."""
        path = event.get("thumbnail_path")
        if path and os.path.isfile(path):
            try:
                # If it's in the LPR evidence dir or cache thumbnails dir, map it
                # For LPR it looks like evidence/lpr/...
                if "evidence/lpr" in path:
                    import urllib.parse
                    # Find everything after evidence/lpr/
                    parts = path.split("evidence/lpr/")
                    if len(parts) > 1:
                        rel = parts[-1]
                        # URL encode any spaces
                        rel = urllib.parse.quote(rel)
                        event["thumbnail"] = f"/recorded/evidence/lpr/{rel}"
                elif "cache/thumbnails" in path:
                    import urllib.parse
                    parts = path.split("cache/thumbnails/")
                    if len(parts) > 1:
                        rel = parts[-1]
                        rel = urllib.parse.quote(rel)
                        event["thumbnail"] = f"/recorded/evidence/cache/{rel}"
                    # Fallback to base64 if it's somewhere else
                    import base64 as b64
                    with open(path, "rb") as f:
                        raw = f.read()
                    event["thumbnail"] = "data:image/jpeg;base64," + b64.b64encode(raw).decode("utf-8")
            except Exception:
                pass
        return event


def _cleanup_file(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _copy2(src, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)
    return True
