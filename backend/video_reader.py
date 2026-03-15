import os
import re
import json
import logging
import config
from fractions import Fraction

try:
    import av
except ImportError:
    av = None
    AVError = Exception
else:
    try:
        AVError = av.AVError
    except Exception:
        try:
            from av.error import Error as AVError
        except Exception:
            AVError = Exception

import cv2

config.configure_logging()
logger = logging.getLogger(__name__)


class VideoReader:
    def __init__(self, path):
        self.path = path
        self._use_av = av is not None
        self._cap = None
        self._dir_mode = False
        self._dir_frames = []
        self._dir_buffer = None
        self._container = None
        self._stream = None
        self._frame_iter = None
        self._last_frame = None
        self._last_time_sec = None
        self._frame_index = -1
        self._seek_target_frame = None
        self._fps = None
        self._frame_count = None
        self._opened = False
        self._cv_buffer = None
        self._open()

    def _create_opencv_cap(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        return cap

    def _apply_opencv_meta(self, cap, start_frame=None):
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps:
            self._fps = float(fps)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count > 0:
            self._frame_count = count
        if start_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    def _switch_to_opencv(self, start_frame=None, reason=None):
        cap = self._create_opencv_cap()
        if cap is None:
            return False
        self._apply_opencv_meta(cap, start_frame=start_frame)
        if reason:
            logger.warning("Switching to OpenCV (%s): %s", reason, self.path)
        self._cleanup_av()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = cap
        self._use_av = False
        self._opened = True
        self._cv_buffer = None
        logger.info("VideoReader opened with OpenCV: %s", self.path)
        return True

    def _should_try_fallback(self):
        if self._frame_count is None:
            return True
        return self._frame_index + 1 < self._frame_count

    def _open(self):
        if os.path.isdir(self.path):
            self._open_directory()
            return
        if self._use_av:
            try:
                self._container = av.open(self.path)
                if not self._container.streams.video:
                    raise RuntimeError("No video stream")
                self._stream = self._container.streams.video[0]
                self._stream.thread_type = "AUTO"
                self._fps = self._calc_fps()
                self._frame_count = self._calc_frame_count()
                self._frame_iter = self._container.decode(self._stream)
                self._opened = True
                logger.info("VideoReader opened with PyAV: %s", self.path)
                return
            except Exception as exc:
                logger.warning("PyAV open failed for %s: %s. Falling back to OpenCV.", self.path, exc)
                self._cleanup_av()
                self._use_av = False
        if not self._switch_to_opencv():
            self._opened = False
            logger.error("VideoReader failed to open: %s", self.path)

    def _open_directory(self):
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        frames = [
            os.path.join(self.path, name)
            for name in sorted(os.listdir(self.path))
            if os.path.splitext(name)[1].lower() in exts
        ]
        if not frames:
            self._opened = False
            logger.error("VideoReader directory empty: %s", self.path)
            return
        meta_path = os.path.join(self.path, "meta.json")
        fps = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                fps = meta.get("target_fps") or meta.get("source_fps")
            except Exception:
                fps = None
        if not fps:
            fps = getattr(config, "FPS", 5)
        self._dir_mode = True
        self._dir_frames = frames
        
        # Build a mapping of original frame_idx to array index if possible
        self._dir_frame_indices = []
        for name in frames:
            # e.g. frame_000724_idx_2896_...
            idx_match = re.search(r"_idx_(\d+)_", name)
            if idx_match:
                self._dir_frame_indices.append(int(idx_match.group(1)))
            else:
                self._dir_frame_indices.append(None)
                
        self._fps = float(fps)
        self._frame_count = len(frames)
        self._frame_index = -1
        self._opened = True
        logger.info("VideoReader opened directory: %s frames=%s", self.path, len(frames))

    def _cleanup_av(self):
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = None
        self._stream = None
        self._frame_iter = None
        self._last_frame = None

    def isOpened(self):
        return bool(self._opened)

    def release(self):
        if self._dir_mode:
            self._opened = False
            self._dir_frames = []
            self._dir_buffer = None
            logger.debug("VideoReader released (directory): %s", self.path)
            return
        if self._use_av:
            self._cleanup_av()
        else:
            if self._cap is not None:
                self._cap.release()
        self._opened = False
        logger.debug("VideoReader released: %s", self.path)

    def _calc_fps(self):
        if not self._stream:
            return None
        rate = None
        if getattr(self._stream, "average_rate", None):
            rate = self._stream.average_rate
        elif getattr(self._stream, "base_rate", None):
            rate = self._stream.base_rate
        elif getattr(self._stream, "guessed_rate", None):
            rate = self._stream.guessed_rate
        if rate:
            try:
                return float(rate)
            except Exception:
                pass
        return None

    def _calc_frame_count(self):
        if not self._stream:
            return None
        if getattr(self._stream, "frames", None):
            if self._stream.frames and self._stream.frames > 0:
                return int(self._stream.frames)
        fps = self._fps
        duration = None
        if self._stream.duration is not None and self._stream.time_base is not None:
            duration = float(self._stream.duration * self._stream.time_base)
        elif self._container is not None and self._container.duration is not None:
            try:
                duration = float(self._container.duration / av.time_base)
            except Exception:
                duration = None
        if duration is not None and fps:
            return int(duration * fps + 0.5)
        return None

    def get(self, prop_id):
        if self._dir_mode:
            if prop_id == cv2.CAP_PROP_FPS:
                return float(self._fps or 0.0)
            if prop_id == cv2.CAP_PROP_FRAME_COUNT:
                return float(self._frame_count or 0.0)
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                return float(max(self._frame_index, 0))
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                if self._last_time_sec is None:
                    return 0.0
                return float(self._last_time_sec * 1000.0)
            return 0.0
        if not self._use_av:
            return self._cap.get(prop_id)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self._fps or 0.0)
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count or 0.0)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return float(max(self._frame_index, 0))
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            if self._last_time_sec is None:
                return 0.0
            return float(self._last_time_sec * 1000.0)
        return 0.0

    def set(self, prop_id, value):
        if self._dir_mode:
            target_idx = None
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                if not self._fps:
                    return False
                target_idx = int(round((float(value) / 1000.0) * float(self._fps)))
            elif prop_id == cv2.CAP_PROP_POS_FRAMES:
                target_idx = int(value)
                
            if target_idx is not None:
                if hasattr(self, "_dir_frame_indices") and self._dir_frame_indices and self._dir_frame_indices[0] is not None:
                    import bisect
                    pos = bisect.bisect_left(self._dir_frame_indices, target_idx)
                    if pos < len(self._dir_frame_indices) and self._dir_frame_indices[pos] == target_idx:
                        self._frame_index = pos - 1
                    else:
                        if pos == 0:
                            self._frame_index = -1
                        elif pos == len(self._dir_frame_indices):
                            self._frame_index = len(self._dir_frame_indices) - 2
                        else:
                            before = self._dir_frame_indices[pos - 1]
                            after = self._dir_frame_indices[pos]
                            if after - target_idx < target_idx - before:
                                self._frame_index = pos - 1
                            else:
                                self._frame_index = pos - 2
                else:
                    self._frame_index = max(-1, target_idx - 1)
                return True
            return False
        if not self._use_av:
            return self._cap.set(prop_id, value)
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return self._seek_time(float(value) / 1000.0)
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            return self._seek_frame(int(value))
        return False

    def _seek_time(self, seconds):
        if not self._container or not self._stream:
            return False
        try:
            time_base = self._stream.time_base
            if not time_base:
                return False
            target = int(seconds / time_base)
            self._container.seek(target, stream=self._stream, any_frame=False, backward=True)
            self._frame_iter = self._container.decode(self._stream)
            self._last_frame = None
            self._last_time_sec = None
            if self._fps:
                self._frame_index = int(seconds * self._fps) - 1
            else:
                self._frame_index = -1
            self._seek_target_frame = None
            logger.debug("VideoReader seek time: %.3fs", seconds)
            return True
        except Exception:
            return False

    def _seek_frame(self, frame_idx):
        if frame_idx < 0:
            frame_idx = 0
        if self._fps:
            seconds = frame_idx / self._fps
        else:
            seconds = 0.0
        ok = self._seek_time(seconds)
        if ok:
            self._seek_target_frame = frame_idx
            logger.debug("VideoReader seek frame: %s", frame_idx)
        return ok

    def _grab_next(self):
        if not self._frame_iter:
            self._frame_iter = self._container.decode(self._stream)
        while True:
            try:
                frame = next(self._frame_iter)
            except StopIteration:
                if self._should_try_fallback():
                    start_frame = self._frame_index + 1 if self._frame_index >= 0 else None
                    if self._switch_to_opencv(start_frame=start_frame, reason="PyAV EOF"):
                        return self._cap.grab()
                return False
            except AVError as exc:
                logger.warning("PyAV decode error for %s: %s. Skipping packet.", self.path, exc)
                if self._should_try_fallback():
                    start_frame = self._frame_index + 1 if self._frame_index >= 0 else None
                    if self._switch_to_opencv(start_frame=start_frame, reason="PyAV decode error"):
                        return self._cap.grab()
                continue
            else:
                break
        self._last_frame = frame
        frame_time = getattr(frame, "time", None)
        if frame_time is None and frame.pts is not None and self._stream and self._stream.time_base:
            frame_time = float(frame.pts * self._stream.time_base)
        self._last_time_sec = frame_time
        if self._fps and frame_time is not None:
            self._frame_index = int(round(frame_time * self._fps))
        else:
            self._frame_index += 1
        return True

    def grab(self):
        if self._dir_mode:
            next_index = self._frame_index + 1
            if next_index >= len(self._dir_frames):
                return False
            frame = cv2.imread(self._dir_frames[next_index])
            if frame is None:
                return False
            self._frame_index = next_index
            self._last_time_sec = (next_index / float(self._fps or 1.0))
            self._dir_buffer = frame
            return True
        if not self._use_av:
            self._cv_buffer = None
            ok = self._cap.grab()
            if ok:
                return True
            ret, frame = self._cap.read()
            if ret:
                self._cv_buffer = frame
                return True
            return False
        if not self._opened:
            return False
        if self._seek_target_frame is not None:
            while True:
                ok = self._grab_next()
                if not ok:
                    self._seek_target_frame = None
                    return False
                if self._frame_index >= self._seek_target_frame:
                    self._seek_target_frame = None
                    return True
        return self._grab_next()

    def retrieve(self):
        if self._dir_mode:
            if self._dir_buffer is None:
                return False, None
            frame = self._dir_buffer
            self._dir_buffer = None
            return True, frame
        if not self._use_av:
            if self._cv_buffer is not None:
                frame = self._cv_buffer
                self._cv_buffer = None
                return True, frame
            return self._cap.retrieve()
        if self._last_frame is None:
            return False, None
        try:
            array = self._last_frame.to_ndarray(format="bgr24")
        except Exception:
            return False, None
        return True, array

    def read(self):
        if self._dir_mode:
            next_index = self._frame_index + 1
            if next_index >= len(self._dir_frames):
                return False, None
            frame = cv2.imread(self._dir_frames[next_index])
            if frame is None:
                return False, None
            self._frame_index = next_index
            self._last_time_sec = (next_index / float(self._fps or 1.0))
            return True, frame
        if not self._use_av:
            return self._cap.read()
        while True:
            ok = self.grab()
            if not ok:
                return False, None
            ret, frame = self.retrieve()
            if ret:
                return True, frame
