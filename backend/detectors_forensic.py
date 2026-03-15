import time
import json
import logging
import re
import base64
import cv2
import numpy as np
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import deque
from video_reader import VideoReader

try:
    import supervision as sv
    from ultralytics import YOLO
    try:
        from ultralytics import YOLOE
    except Exception:
        YOLOE = None
    import torch
    import torchvision.transforms as T
except ImportError:
    torch = None
    YOLOE = None
    pass

# Config
import config

config.configure_logging()
logger = logging.getLogger(__name__)

# #region agent log
# _DEBUG_LOG_PATH = "/home/username/Documents/Project-u/.cursor/debug-c61cc6.log"
# def _debug_log(msg, data, hypothesis_id=None, location=""):
#     try:
#         payload = {"sessionId": "c61cc6", "timestamp": int(time.time() * 1000), "location": location, "message": msg, "data": data}
#         if hypothesis_id:
#             payload["hypothesisId"] = hypothesis_id
#         with open(_DEBUG_LOG_PATH, "a") as f:
#             f.write(json.dumps(payload) + "\n")
#     except Exception:
#         pass
# #endregion

def _encode_thumbnail(frame):
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    try:
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            return None
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None

def _is_cancel_requested(cancel_check):
    if not callable(cancel_check):
        return False
    try:
        return bool(cancel_check())
    except Exception as exc:
        logger.warning("Cancel check failed: %s", exc)
        return False


class LicensePlateSkill:
    """
    Refactored from number_plate.py
    Identifies vehicles, tracks them (BoT-SORT), and extracts license plates using VLM/OCR.
    """
    def __init__(self, model_path="yoloe-26x-seg.pt", plate_model_path="best.pt"):
        logger.info("Initializing LicensePlateSkill...")
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        self.use_half = False # Disabled FP16 as per request
        self.max_ocr_crops_per_track = 4
        self.open_vocab = False
        self.vehicle_prompts = ["car", "motorcycle", "bus", "truck", "suv", "van", "rickshaw", "vehicle"]
        self.lpr_prompts = self.vehicle_prompts

        # 1. Load Object Detection Model (Vehicles)
        try:
            if YOLOE is None:
                raise RuntimeError("YOLOE not available in ultralytics")
            self.model = YOLOE(model_path)
            self.open_vocab = False
            # YOLOE is open-vocab and REQUIRES set_classes() to detect anything.
            # Use a broad vehicle list to maximize coverage.
            if hasattr(self.model, "set_classes"):
                self.model.set_classes(self.vehicle_prompts)
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not load primary model {model_path}: {e}")
            self.model = None

        # 2. Load License Plate Detection Model (LPD)
        try:
            self.plate_model = YOLO(plate_model_path)
            self.plate_model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not load plate model {plate_model_path}: {e}")
            self.plate_model = None
            
        self.class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        # 3. VLM for OCR
        from detectors import VLM_Lite
        self.vlm = VLM_Lite()
        self.track_best = {}

    def _normalize_plate_text(self, text):
        if text is None:
            return None
        raw = str(text).upper().strip()
        cleaned = re.sub(r"[^A-Z0-9]", "", raw)
        if len(cleaned) < 3:
            return None
        return cleaned

    def _frame_hash(self, frame):
        """
        Compute a simple perceptual hash for frame deduplication.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            mean = float(small.mean())
            bits = (small > mean).astype(np.uint8)
            return bits.tobytes()
        except Exception:
            return None

    def _expand_plate_variants(self, text, max_variants=64):
        normalized = self._normalize_plate_text(text)
        if not normalized:
            return set()

        confusion_map = {
            "0": "O", "O": "0", "1": "I", "I": "1",
            "5": "S", "S": "5", "8": "B", "B": "8",
            "2": "Z", "Z": "2", "6": "G", "G": "6",
        }
        options = []
        for ch in normalized:
            alt = confusion_map.get(ch)
            if alt and alt != ch:
                options.append((ch, alt))
            else:
                options.append((ch,))

        variants = set()

        def _dfs(idx, parts):
            if len(variants) >= max_variants:
                return
            if idx >= len(options):
                variants.add("".join(parts))
                return
            for candidate in options[idx]:
                parts.append(candidate)
                _dfs(idx + 1, parts)
                parts.pop()

        _dfs(0, [])
        return variants or {normalized}

    def _levenshtein_distance(self, a, b, max_distance=2):
        if a == b:
            return 0
        if abs(len(a) - len(b)) > max_distance:
            return max_distance + 1

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current = [i]
            row_min = i
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                value = min(
                    prev[j] + 1,         # deletion
                    current[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,  # substitution
                )
                current.append(value)
                if value < row_min:
                    row_min = value
            if row_min > max_distance:
                return max_distance + 1
            prev = current
        return prev[-1]

    def _plate_matches_target(self, plate_text, target_plate_hint):
        if not target_plate_hint:
            return True

        plate = self._normalize_plate_text(plate_text)
        target = self._normalize_plate_text(target_plate_hint)
        if not plate or not target:
            return False

        if target in plate:
            return True
        if plate in target and (len(plate) / float(max(1, len(target)))) >= 0.75:
            return True

        plate_variants = self._expand_plate_variants(plate)
        target_variants = self._expand_plate_variants(target)
        for p_variant in plate_variants:
            for t_variant in target_variants:
                if t_variant in p_variant:
                    return True
                min_len = min(len(p_variant), len(t_variant))
                max_dist = 1 if min_len <= 10 else 2
                if self._levenshtein_distance(p_variant, t_variant, max_distance=max_dist) <= max_dist:
                    return True

        return False

    def _detect_plate_in_crop(self, vehicle_crop):
        """
        Runs the LPD model on a specific vehicle crop.
        Returns: (plate_crop, xyxy_relative_box, confidence)
        """
        if self.plate_model is None or vehicle_crop is None or vehicle_crop.size == 0:
            return None, None, 0.0

        results = self.plate_model.predict(vehicle_crop, verbose=False, conf=0.25, iou=0.7)
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            return None, None, 0.0

        h_img, w_img = vehicle_crop.shape[:2]
        boxes = results[0].boxes
        candidates = []
        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            conf = float(boxes.conf[i])
            # Bottom-center of the plate box
            center_y = (y1 + y2) / 2.0
            candidates.append((center_y, conf, x1, y1, x2, y2))

        if not candidates:
            return None, None, 0.0

        # Pick the bottom-most plate (highest center_y = closest to bumper)
        candidates.sort(key=lambda c: c[0], reverse=True)
        center_y, conf, x1, y1, x2, y2 = candidates[0]

        # Pad for better OCR readability
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.15)
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w_img, x2 + pad_x)
        py2 = min(h_img, y2 + pad_y)

        # Return just the box coordinates (we will draw on full image instead of cropping)
        return None, [px1, py1, px2, py2], conf

    def _detect_plate_box_in_vehicle(self, frame, vehicle_bbox):
        """
        Runs the LPD model inside a vehicle box, returns best plate box in full-frame coords.
        """
        if self.plate_model is None or frame is None or frame.size == 0:
            return None, None
        if not vehicle_bbox or len(vehicle_bbox) != 4:
            return None, None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        vehicle_crop = frame[y1:y2, x1:x2]
        results = self.plate_model.predict(vehicle_crop, verbose=False, conf=0.25, iou=0.7)
        if not results or not results[0].boxes:
            return None, None
        boxes_obj = results[0].boxes
        boxes = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        bx1, by1, bx2, by2 = [int(v) for v in boxes[best_idx]]
        # Map to full-frame coords
        full_box = [x1 + bx1, y1 + by1, x1 + bx2, y1 + by2]
        return full_box, float(confs[best_idx])

    def _yoloe_vehicle_bboxes(self, frame):
        if self.model is None or frame is None or frame.size == 0:
            return []
        try:
            results = self.model.predict(frame, verbose=False, classes=[2, 3, 5, 7], conf=0.1)
            if not results or not results[0].boxes:
                return []
            boxes = results[0].boxes.xyxy.cpu().numpy()
            return [[int(v) for v in box] for box in boxes]
        except Exception:
            return []

    def _crop_quality_score(self, plate_crop, vehicle_dims, plate_conf):
        """
        Refined scoring for direct plate crops.
        """
        if plate_crop is None or plate_crop.size == 0:
            return 0.0
        
        ph, pw = plate_crop.shape[:2]
        
        # Too small to read
        if ph < 15 or pw < 30: 
            return 0.0

        # Sharpness check
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Weighted Score: Resolution + Confidence + Sharpness
        res_score = min(ph, 100) / 100.0 
        sharp_score = min(sharpness, 500) / 500.0
        
        return (res_score * 0.4) + (plate_conf * 0.4) + (sharp_score * 0.2)

    def _vehicle_crop_score(self, vehicle_crop, bbox=None, frame_shape=None):
        """
        Returns a dict with two separate scores:
          - area_score:        Effective crop area (penalized if clipped at edges).
          - readability_score: Composite clarity score (sharpness, contrast, brightness, plate).
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return {"area_score": 0, "readability_score": 0.0}

        h, w = vehicle_crop.shape[:2]
        if h < 24 or w < 24:
            return {"area_score": 0, "readability_score": 0.0}

        # ---- SCORE 1: Area (raw pixel area, penalized if clipped) ----
        raw_area = h * w
        
        edge_penalty = 1.0
        EDGE_MARGIN = 5
        if bbox is not None and frame_shape is not None:
            bx1, by1, bx2, by2 = bbox
            fh, fw = frame_shape[:2]
            touches = sum([
                bx1 <= EDGE_MARGIN,
                by1 <= EDGE_MARGIN,
                bx2 >= (fw - EDGE_MARGIN),
                by2 >= (fh - EDGE_MARGIN),
            ])
            if touches == 1:
                edge_penalty = 0.5
            elif touches >= 2:
                edge_penalty = 0.2
        
        # Aspect ratio sanity: very elongated boxes are likely partial/occluded
        aspect = w / max(h, 1)
        if aspect > 5.0 or aspect < 0.15:
            edge_penalty *= 0.3  # Very unusual shape, likely not a proper vehicle view
        
        area_score = int(raw_area * edge_penalty)
        
        # ---- SCORE 2: Readability (multi-factor) ----
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # (a) Sharpness via Laplacian variance — higher = sharper/less blurry
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_score = min(sharpness, 500) / 500.0  # 0.0 to 1.0
        
        # (b) Contrast via StdDev of pixel intensities — low = washed out or uniform
        contrast = float(np.std(gray))
        contrast_score = min(contrast, 60) / 60.0  # 0.0 to 1.0 (60 is good contrast)
        
        # (c) Brightness check — penalize too dark or overexposed
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 40:
            brightness_score = mean_brightness / 40.0  # Very dark: 0.0-1.0
        elif mean_brightness > 220:
            brightness_score = (255 - mean_brightness) / 35.0  # Overexposed: 0.0-1.0
        else:
            brightness_score = 1.0  # Good range
        
        # (d) Plate presence bonus
        plate_bonus = 0.0
        if self.plate_model:
            try:
                res = self.plate_model(vehicle_crop, verbose=False, conf=0.15)
                if res and res[0].boxes and len(res[0].boxes) > 0:
                    confs = res[0].boxes.conf.cpu().numpy()
                    best_conf = float(np.max(confs))
                    plate_bonus = best_conf * 0.7  # Up to +0.7
            except Exception:
                pass
        
        # Weighted composite: sharpness matters most, then contrast, then brightness
        readability_score = (
            sharp_score * 0.25 +
            contrast_score * 0.2 +
            brightness_score * 0.1 +
            plate_bonus  # Up to +0.7 bonus
        )  # Range: 0.0 to ~1.25
        
        return {"area_score": area_score, "readability_score": round(readability_score, 4)}

    def _is_visual_duplicate(self, new_crop, last_sample, threshold=15.0):
        """
        Check if new_crop is visually similar to the LAST captured sample.
        Returns: (is_duplicate, matching_sample_dict)
        """
        if new_crop is None or new_crop.size == 0:
            return True, None
        
        # If no last sample, not a duplicate
        if not last_sample:
            return False, None

        try:
            new_gray = cv2.cvtColor(new_crop, cv2.COLOR_BGR2GRAY)
            new_small = cv2.resize(new_gray, (32, 32)).astype(np.float32)
            
            old_crop = last_sample.get("crop")
            if old_crop is not None:
                old_gray = cv2.cvtColor(old_crop, cv2.COLOR_BGR2GRAY)
                old_small = cv2.resize(old_gray, (32, 32)).astype(np.float32)
                
                diff = np.mean(np.abs(new_small - old_small))
                if diff < threshold:
                    return True, last_sample
        except Exception:
            pass
            
        return False, None

    def _extract_target_plate_hint(self, text):
        if not text:
            return None
        upper = str(text).upper()
        tokens = re.findall(r"[A-Z0-9]{2,}", upper)
        def _has_alpha(t):
            return any(ch.isalpha() for ch in t)

        def _has_digit(t):
            return any(ch.isdigit() for ch in t)

        candidates = []
        for token in tokens:
            if _has_alpha(token) and _has_digit(token):
                candidates.append(token)

        for idx in range(len(tokens) - 1):
            left = tokens[idx]
            right = tokens[idx + 1]
            if not (_has_alpha(left) and _has_digit(left)):
                continue
            if not (_has_digit(right)):
                continue
            merged = left + right
            if 4 <= len(merged) <= 14:
                candidates.append(merged)

        best = None
        best_score = None
        for cand in candidates:
            normalized = self._normalize_plate_text(cand)
            if not normalized:
                continue
            nlen = len(normalized)
            if nlen > 14:
                continue
            length_score = 2 if 6 <= nlen <= 11 else 1
            score = (length_score, nlen)
            if best is None or score > best_score:
                best = normalized
                best_score = score

        return best

    def _encode_ocr_input(self, crop, *, allow_resize=True, allow_sharpen=False):
        """
        Prepares crop for VLM/OCR API.
        UPDATED: High quality, no forced resizing/sharpening by default to prevent degradation.
        """
        if crop is None or crop.size == 0:
            return None, None
        # #region agent log
        # h0, w0 = crop.shape[:2] if crop is not None else (0, 0)
        # #endregion

        # Only upscale if TINY (unreadable otherwise), but don't downscale
        if allow_resize:
            h, w = crop.shape[:2]
            if h < 32:
                scale = 64 / h
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # #region agent log
        # h1, w1 = crop.shape[:2] if crop is not None else (0, 0)
        # _debug_log("encode_ocr_input", {"allow_resize": allow_resize, "shape_before": [h0, w0], "shape_after": [h1, w1], "resized": (h0, w0) != (h1, w1)}, "A", "detectors_forensic.py:_encode_ocr_input")
        # #endregion

        if allow_sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            crop = cv2.filter2D(crop, -1, kernel)

        # High Quality JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        ret, buffer = cv2.imencode(".jpg", crop, encode_params)
        if not ret:
            return None, None
        return crop, buffer

    def _build_ocr_collage(self, crops, cols=2):

        if len(crops) == 1:
            return crops[0]

        # Filter valid crops
        valid = [c for c in crops if c is not None and getattr(c, "size", 0) > 0]
        if not valid:
            return None

        n = len(valid)
        cols = max(1, min(cols, n))
        rows = (n + cols - 1) // cols

        cell_w = max(c.shape[1] for c in valid)
        cell_h = max(c.shape[0] for c in valid)

        collage = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)

        for idx, crop in enumerate(valid):
            r = idx // cols
            c = idx % cols
            ch, cw = crop.shape[:2]
            # Center original crop in cell (no resize)
            y_off = (cell_h - ch) // 2
            x_off = (cell_w - cw) // 2
            y1 = r * cell_h + y_off
            x1 = c * cell_w + x_off
            collage[y1:y1 + ch, x1:x1 + cw] = crop
            # Draw panel number label
            label_x = c * cell_w + 5
            label_y = r * cell_h + 25
            cv2.putText(collage, str(idx + 1), (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # Draw thin border around the cell
            cv2.rectangle(collage,
                          (c * cell_w, r * cell_h),
                          ((c + 1) * cell_w - 1, (r + 1) * cell_h - 1),
                          (80, 80, 80), 1)

        return collage

    def _extract_plate_with_vlm_collage(self, collage_img, n_panels=1):
        if collage_img is None or getattr(collage_img, "size", 0) == 0:
            return None, {"error": "empty_collage"}

        # Encode the collage as JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        ret, buffer = cv2.imencode(".jpg", collage_img, encode_params)
        if not ret:
            return None, {"error": "encoding_failed"}

        if n_panels <= 1:
            prompt_text = (
                "This is a cropped image of a single vehicle. "
                "Read the license plate number visible on this vehicle. "
                "The plate may be single-line or two-line. "
                "For two-line plates, read top line then bottom line and combine them. "
                "Return strict JSON only: {'plates': ['PLATE1']}. "
                "If no valid plate is visible return {'plates': []}."
            )
        else:
            prompt_text = (
                f"This collage shows {n_panels} numbered panels of the SAME vehicle "
                "captured from different frames of a video. "
                "Each panel may show the license plate from a slightly different angle or distance. "
                "Carefully read the license plate text visible in EACH panel. "
                "The plate may be single-line or two-line. "
                "For two-line plates, read top line then bottom line and combine them. "
                "Compare your readings across all panels and determine the SINGLE BEST "
                "(most confident and consistent) plate number. "
                "Return strict JSON only: {'plates': ['BEST_PLATE']}. "
                "If no valid plate is visible in any panel return {'plates': []}."
            )

        files = {"file": ("collage.jpg", buffer.tobytes(), "image/jpeg")}

        try:
            response = self.vlm.session.post(
                self.vlm.endpoint,
                files=files,
                data={"prompt": prompt_text},
                timeout=8,
            )
            if response.status_code == 200:
                try:
                    res = response.json()
                    content = res
                    if "choices" in res:
                        content = res["choices"][0]["message"]["content"]
                        if isinstance(content, str):
                            content = content.replace("```json", "").replace("```", "").strip()
                            try:
                                content = json.loads(content)
                            except Exception:
                                content = {"plate": content}

                    plates = content.get("plates")
                    if plates is None:
                        single = content.get("plate") or content.get("plate_text") or content.get("text")
                        plates = [single] if single else []
                    if not isinstance(plates, list):
                        plates = [plates]
                    normalized = [self._normalize_plate_text(p) for p in plates]
                    normalized = [p for p in normalized if p]
                    return normalized, res
                except Exception as e:
                    return None, {"error": "parsing_failed", "details": str(e)}
            return None, {"error": f"http_{response.status_code}"}
        except Exception as e:
            return None, {"error": str(e)}


    def _extract_plate_with_vlm(self, plate_crop, crop_kind=None, ocr_input=None, bboxes=None):
        """Sends vehicle crop to VLM for plate reading."""
        # Use provided buffer if available to avoid re-encoding
        if ocr_input:
            _, buffer = ocr_input
        else:
            _, buffer = self._encode_ocr_input(plate_crop)
            
        if buffer is None:
            return None, {"error": "encoding_failed"}

        prompt_text = (
            "This is a cropped image of a single vehicle. "
            "Read the license plate number visible on this vehicle. "
             "The plate may be single-line or two-line. "
            "For two-line plates, read top line then bottom line and combine them. "
            "Return strict JSON only: {'plates': ['PLATE1'], 'best_bbox': null}. "
            "If no valid plate is visible return {'plates': [], 'best_bbox': null}."
        )
        files = {"file": ("plate.jpg", buffer.tobytes(), "image/jpeg")}

        try:
            response = self.vlm.session.post(
                self.vlm.endpoint,
                files=files,
                data={"prompt": prompt_text},
                timeout=8,
            )
            if response.status_code == 200:
                try:
                    res = response.json()
                    content = res
                    if "choices" in res:
                        content = res["choices"][0]["message"]["content"]
                        if isinstance(content, str):
                            content = content.replace("```json", "").replace("```", "").strip()
                            try:
                                content = json.loads(content)
                            except:
                                content = {"plate": content} 
                    
                    plates = content.get("plates")
                    if plates is None:
                        single = content.get("plate") or content.get("plate_text") or content.get("text")
                        plates = [single] if single else []
                    if not isinstance(plates, list):
                        plates = [plates]
                    normalized = [self._normalize_plate_text(p) for p in plates]
                    normalized = [p for p in normalized if p]
                    best_bbox = content.get("best_bbox") or content.get("bbox") or content.get("best_box")
                    if isinstance(best_bbox, (list, tuple)) and len(best_bbox) == 4:
                        try:
                            best_bbox = [int(float(v)) for v in best_bbox]
                        except Exception:
                            best_bbox = None
                    else:
                        best_bbox = None
                    best_idx = content.get("best_box_index")
                    try:
                        if best_idx is not None:
                            best_idx = int(best_idx)
                    except Exception:
                        best_idx = None
                    return normalized, res, best_bbox, best_idx
                except Exception as e:
                    return None, {"error": "parsing_failed", "details": str(e)}, None, None
            return None, {"error": f"http_{response.status_code}"}, None, None
        except Exception as e:
            return None, {"error": str(e)}, None, None

    def process_video(
        self,
        video_path,
        target_plate=None,
        frame_stride=3,
        start_frame=0,
        end_frame=None,
        show_progress=True,
        return_state=False,
        plate_crop_dir="plate_crops",
        save_plate_crops=False,
        plate_frame_dir="plate_frames",
        save_plate_frames=False,
        ocr_debug_dir="ocr_debug",
        save_ocr_debug_crops=False,
        ocr_debug_limit=1000,
        **kwargs,
    ):
        """
        Scans video for vehicles, reads plates, and optionally filters by target_plate.
        """
        if self.model is None:
            logger.error("Vehicle model not loaded correctly. Aborting LPR.")
            return []

        # Favor preprocessed frames for detection phase if available
        input_source = kwargs.get("preprocessed_dir") or video_path
        cap = VideoReader(input_source)
        if not cap.isOpened():
            logger.error("LPR failed to open video source: %s", input_source)
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = getattr(config, "FPS", 30)
        logger.info("LPR scan start: path=%s fps=%s start=%s end=%s stride=%s", video_path, fps, start_frame, end_frame, frame_stride)
        self.track_best = {}
        frame_stride = max(1, int(frame_stride))
        cancel_check = kwargs.get("cancel_check")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

        frame_idx = int(start_frame or 0)
        segment_total = (
            (end_frame - start_frame + 1)
            if end_frame is not None
            else max(0, total_frames - frame_idx)
        )
        pbar = tqdm(
            total=segment_total,
            desc="LPR Scan",
            disable=not show_progress or end_frame is not None,
        )

        # DEBUG: Create/Clear Debug Directories
        debug_base = Path("debug_output")
        debug_dirs = {
            "1_detections": debug_base / "1_detections",
            "2_crops": debug_base / "2_crops",
            "3_vlm_input": debug_base / "3_vlm_input",
            "4_vlm_output": debug_base / "4_vlm_output",
        }
        for d in debug_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        while cap.isOpened():
            if _is_cancel_requested(cancel_check):
                logger.info("LPR scan canceled during frame sweep")
                break
            if end_frame is not None and frame_idx > end_frame:
                break

            grabbed = cap.grab()
            if not grabbed:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            ret, frame = cap.retrieve()
            if not ret:
                logger.warning("LPR retrieve failed at frame %s; skipping.", frame_idx)
                frame_idx += 1
                pbar.update(1)
                continue

            current_idx = frame_idx
            frame_idx += 1
            pbar.update(1)

            logger.debug(f"[DEBUG] Processing Frame {current_idx} (Time: {current_idx / fps:.2f}s)")

            # 1. Detect & Track
            det_results = None
            try:
                track_kwargs = dict(
                    persist=True, verbose=False, tracker="botsort.yaml",
                    conf=0.10,
                    device=self.device,
                )
                if torch:
                    with torch.inference_mode():
                        det_results = self.model.track(frame, **track_kwargs)
                else:
                    det_results = self.model.track(frame, **track_kwargs)
            except Exception as exc:
                logger.warning("LPR track failed, falling back to predict: %s", exc)

            if not det_results:
                logger.debug(f"[DEBUG] Frame {current_idx}: No detection results returned.")
                continue

            boxes_obj = det_results[0].boxes
            if boxes_obj is None:
                logger.debug(f"[DEBUG] Frame {current_idx}: No boxes found.")
                continue

            boxes = boxes_obj.xyxy.cpu().numpy()
            classes = boxes_obj.cls.cpu().numpy().astype(int)
            ids_tensor = getattr(boxes_obj, "id", None)
            if ids_tensor is None or ids_tensor.numel() == 0:
                track_ids = (np.arange(len(boxes), dtype=int) + (current_idx * 1000))
            else:
                track_ids = ids_tensor.cpu().numpy().astype(int)
            if len(track_ids) == 0:
                logger.debug(f"[DEBUG] Frame {current_idx}: 0 track IDs.")
                continue

            # #region agent log
            # _debug_log("yolo_frame_detections", {"frame_idx": int(current_idx), "n_boxes": len(boxes), "n_track_ids": len(track_ids), "track_ids": track_ids.tolist()[:20]}, "B", "detectors_forensic.py:detection_loop")
            # #endregion

            # DEBUG Step 1: Save full frame with all boxes
            # DEBUG Step 1: Save full frame with all boxes
            if len(track_ids) > 0:
                debug_frame = frame.copy() # Use copy
                for i, tid in enumerate(track_ids):
                     x1, y1, x2, y2 = map(int, boxes[i])
                     cls_obj = classes[i]
                     cls_name = str(self.model.names[cls_obj])
                     cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                     cv2.putText(debug_frame, f"{tid}:{cls_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite(str(debug_dirs["1_detections"] / f"frame_{current_idx}.jpg"), debug_frame)
            
            for i, tid in enumerate(track_ids):
                if _is_cancel_requested(cancel_check):
                    logger.info("LPR scan canceled during track collection")
                    break
                x1, y1, x2, y2 = map(int, boxes[i])
                class_label = str(self.model.names[classes[i]])
                
                logger.debug(f"[DEBUG] Raw Detection: TID={tid} Class={class_label} ({classes[i]})")

                # Fallback mapping if standard COCO
                if class_label.isdigit() and int(class_label) in self.class_names:
                     class_label = self.class_names[int(class_label)]

                if class_label.lower() not in self.vehicle_prompts:
                    # UPDATED: Allow all tracks, but log if interesting
                    # continue 
                    pass

                # Check for reasonable vehicle size
                h_img, w_img = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                # Expand bounding box to ensure plate is not cut off at the bottom
                box_w = x2 - x1
                box_h = y2 - y1
                
                nx1 = max(0, x1 - int(box_w * 0.05))
                nx2 = min(w_img, x2 + int(box_w * 0.05))
                ny1 = max(0, y1 - int(box_h * 0.05))
                ny2 = min(h_img, y2 + int(box_h * 0.15)) # 15% downward expansion
                
                # Extract Vehicle Crop with expanded bounds
                vehicle_crop = frame[ny1:ny2, nx1:nx2]

                # Calculate two-stage score: area (fullness) + readability (clarity)
                scores = self._vehicle_crop_score(vehicle_crop, bbox=[x1, y1, x2, y2], frame_shape=frame.shape)
                area_score = scores["area_score"]
                readability_score = scores["readability_score"]

                # Store result (use full frame for VLM)
                entry = self.track_best.setdefault(
                    tid,
                    {
                        "class": class_label,
                        "samples": [],
                    },
                )
                if len(entry["samples"]) == 0:
                     logger.info(f"New Track Added: ID={tid} Class={class_label}")
                entry["class"] = class_label

                # Check for visual duplicates with LAST sample only
                last_sample = entry["samples"][-1] if entry["samples"] else None
                is_dup, dup_sample = self._is_visual_duplicate(vehicle_crop, last_sample)
                
                sample_data = {
                    "frame_idx": current_idx,
                    "crop": frame.copy(),
                    "vehicle_crop": vehicle_crop.copy() if vehicle_crop.size > 0 else None,
                    "thumbnail": None,
                    "timestamp": current_idx / fps,
                    "area_score": area_score,
                    "readability_score": readability_score,
                    "score": area_score,  # backward compat
                    "crop_kind": "frame",
                    "bbox": [nx1, ny1, nx2, ny2],
                    "vehicle_bbox": [nx1, ny1, nx2, ny2],
                }

                if is_dup:
                    # Replace duplicate only if new crop has bigger area or same area but more readable
                    old_area = dup_sample.get("area_score", 0)
                    old_read = dup_sample.get("readability_score", 0.0)
                    if area_score > old_area or (area_score >= old_area and readability_score > old_read + 0.05):
                        dup_sample.update(sample_data)
                else:
                    entry["samples"].append(sample_data)

                # Pruning: keep best by area first, then readability
                if len(entry["samples"]) > (self.max_ocr_crops_per_track * 3):
                     entry["samples"].sort(
                         key=lambda s: (s.get("area_score", 0), s.get("readability_score", 0.0)),
                         reverse=True
                     )
                     del entry["samples"][self.max_ocr_crops_per_track * 2:]
            if _is_cancel_requested(cancel_check):
                break

        cap.release()
        pbar.close()

        # #region agent log
        # _debug_log("track_best_after_detection", {"n_tracks": len(self.track_best), "track_ids": list(self.track_best.keys())[:50], "samples_per_track": {str(tid): len(d.get("samples") or []) for tid, d in list(self.track_best.items())[:20]}}, "B", "detectors_forensic.py:after_detection_loop")
        # #endregion

        # Post-Processing: Two-stage selection per track
        # Stage 1: Pick crops with LARGEST area (= most of the vehicle visible)
        # Stage 2: Among those, pick the most READABLE (sharpest / plate visible)
        for tid, data in self.track_best.items():
            if _is_cancel_requested(cancel_check):
                logger.info("LPR scan canceled during sample selection")
                break
            samples = data.get("samples") or []
            if not samples:
                continue
            
            # Stage 1: Sort by area_score DESC, find the max-area tier
            samples.sort(key=lambda x: x.get("area_score", 0), reverse=True)
            max_area = samples[0].get("area_score", 0)
            if max_area == 0:
                continue
            
            # Keep all samples within 20% of max area (= similarly full vehicles)
            area_threshold = max_area * 0.7
            big_crops = [s for s in samples if s.get("area_score", 0) >= area_threshold]
            
            # Stage 2: Among big crops, sort by readability DESC
            big_crops.sort(key=lambda x: x.get("readability_score", 0.0), reverse=True)
            
            # Take top N most readable from the big-area set
            selected_samples = big_crops[:self.max_ocr_crops_per_track]

            # #region agent log
            # _debug_log("selected_samples_per_track", {"tid": int(tid), "n_samples_before": len(samples), "n_selected": len(selected_samples)}, "C", "detectors_forensic.py:post_process_select")
            # #endregion

            # 2. Swap crop from full frame -> vehicle crop for VLM
            for s in selected_samples:
                v_crop = s.get("vehicle_crop")
                if v_crop is not None and v_crop.size > 0:
                    s["full_frame"] = s.get("crop")  # Keep full frame for debug/thumbnail
                    s["crop"] = v_crop                # VLM will see the vehicle crop
                    s["crop_kind"] = "vehicle_crop"
                    
                    # Save to 2_crops debug dir
                    score = s.get("score", 0.0)
                    f_idx = s.get("frame_idx")
                    crop_fname = f"crop_t{tid}_f{f_idx}_score{score:.2f}.jpg"
                    cv2.imwrite(str(debug_dirs["2_crops"] / crop_fname), v_crop)
            
            # Replace full list with selected vehicle crops
            data["samples"] = selected_samples

        # Save OCR crops to disk (per-track best samples)
        if save_plate_crops and self.track_best:
            try:
                crop_root = Path(plate_crop_dir)
                crop_root.mkdir(parents=True, exist_ok=True)
                for tid, data in self.track_best.items():
                    if _is_cancel_requested(cancel_check):
                        logger.info("LPR scan canceled while saving crop samples")
                        break
                    samples = data.get("samples") or []
                    for idx, sample in enumerate(samples):
                        if _is_cancel_requested(cancel_check):
                            logger.info("LPR scan canceled while saving crop samples")
                            break
                        crop = sample.get("crop")
                        if crop is None or getattr(crop, "size", 0) == 0:
                            continue
                        frame_idx = int(sample.get("frame_idx", -1))
                        ts = float(sample.get("timestamp", -1.0))
                        fname = f"frame_t{tid}_f{frame_idx}_i{idx}_t{ts:.2f}.jpg"
                        cv2.imwrite(str(crop_root / fname), crop)
            except Exception as exc:
                logger.warning("Failed saving plate crops: %s", exc)

        # 5. Post-Process: Run VLM OCR on Best Crops
        logger.info("Running VLM OCR on %s unique vehicles", len(self.track_best))

        target_plate_hint = self._extract_target_plate_hint(target_plate)
        if target_plate_hint:
            logger.info("LPR target plate hint=%s", target_plate_hint)

        ocr_debug_root = None
        ocr_debug_count = 0
        if save_ocr_debug_crops and ocr_debug_limit and ocr_debug_limit > 0:
            try:
                ocr_debug_root = Path(ocr_debug_dir)
                ocr_debug_root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Failed to create OCR debug dir: %s", exc)
                ocr_debug_root = None

        final_events = []
        plate_frame_root = None
        if save_plate_frames:
            try:
                plate_frame_root = Path(plate_frame_dir)
                plate_frame_root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Failed to create plate frame dir: %s", exc)
                plate_frame_root = None

        def _save_matched_frame(frame_img, tid, frame_idx, ts, plate_value):
            if plate_frame_root is None:
                return
            if frame_img is None or getattr(frame_img, "size", 0) == 0:
                return
            safe_plate = re.sub(r"[^A-Z0-9]+", "_", str(plate_value or "").upper()).strip("_")
            if not safe_plate:
                safe_plate = "UNKNOWN"
            fname = f"plate_match_t{tid}_f{frame_idx}_t{ts:.2f}_{safe_plate}.jpg"
            try:
                cv2.imwrite(str(plate_frame_root / fname), frame_img)
            except Exception as exc:
                logger.warning("Failed saving plate frame: %s", exc)
        seen_frame_hashes = set()
        for tid, data in tqdm(self.track_best.items(), desc="DOCR"):
            if _is_cancel_requested(cancel_check):
                logger.info("LPR scan canceled during OCR stage")
                break
            samples = data.get("samples") or []
            if not samples:
                continue

            # ── Collage-based OCR: collect processed crops for all samples ──
            collage_crops = []
            for sample in samples:
                if _is_cancel_requested(cancel_check):
                    logger.info("LPR scan canceled during OCR stage")
                    break
                ocr_input = None
                ocr_crop = sample.get("crop")  # This is now a vehicle crop
                sample_frame_idx = int(sample.get("frame_idx", -1))

                # VISUAL PROMPTING: detect plate and draw guidance box
                _, plate_box, plate_conf = self._detect_plate_in_crop(ocr_crop)
                vlm_crop = ocr_crop
                if plate_box is not None:
                    vlm_crop = ocr_crop.copy()
                    bx1, by1, bx2, by2 = plate_box
                    cv2.rectangle(vlm_crop, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    logger.debug("Drew VLM guidance box for track %s frame %s (conf=%.2f)",
                                 tid, sample_frame_idx, plate_conf)

                collage_crops.append(vlm_crop)

                # DEBUG Step 3: Save individual VLM inputs
                vlm_input_fname = f"vlm_input_t{tid}_f{sample_frame_idx}.jpg"
                cv2.imwrite(str(debug_dirs["3_vlm_input"] / vlm_input_fname), vlm_crop)

                if ocr_debug_root is not None and ocr_debug_count < ocr_debug_limit:
                    try:
                        fname = f"ocr_t{tid}_f{sample_frame_idx}_{ocr_debug_count}_vehicle_crop.jpg"
                        cv2.imwrite(str(ocr_debug_root / fname), vlm_crop)
                        ocr_debug_count += 1
                    except Exception as exc:
                        logger.warning("Failed saving OCR debug crop: %s", exc)

            if _is_cancel_requested(cancel_check):
                break
            if not collage_crops:
                continue

            # ── Build collage and send single VLM call ──
            collage_img = self._build_ocr_collage(collage_crops, cols=2)
            if collage_img is None:
                continue

            # DEBUG: Save the collage itself
            collage_debug_fname = f"vlm_input_collage_t{tid}.jpg"
            cv2.imwrite(str(debug_dirs["3_vlm_input"] / collage_debug_fname), collage_img)

            logger.info("Track %s: sending collage of %d panels to VLM", tid, len(collage_crops))
            plate_text, vlm_response = self._extract_plate_with_vlm_collage(
                collage_img, n_panels=len(collage_crops)
            )

            if _is_cancel_requested(cancel_check):
                break

            # Use the best sample's metadata for the event
            best_sample = samples[0]
            vehicle_bbox = best_sample.get("vehicle_bbox") or best_sample.get("bbox")
            ocr_crop = best_sample.get("crop")
            full_frame = best_sample.get("full_frame")

            # Generate single-image thumbnail for evidence (use full frame raw)
            if full_frame is not None and full_frame.size > 0:
                evidence_thumbnail = _encode_thumbnail(full_frame)
            else:
                evidence_thumbnail = _encode_thumbnail(ocr_crop)

            if not plate_text:
                logger.debug(f"[DEBUG] VLM Collage Result for Track {tid}: No plates found.")
                vlm_out_fname = f"vlm_output_t{tid}_collage_FAIL.json"
                debug_data = {
                    "track_id": int(tid),
                    "n_panels": len(collage_crops),
                    "plate_text": None,
                    "raw_response": vlm_response
                }
                with open(debug_dirs["4_vlm_output"] / vlm_out_fname, "w") as f:
                    json.dump(debug_data, f, indent=2)
            else:
                # DEBUG Step 4: Save VLM Output (Success)
                vlm_out_fname = f"vlm_output_t{tid}_collage.json"
                debug_data = {
                    "track_id": int(tid),
                    "n_panels": len(collage_crops),
                    "plate_text": plate_text,
                    "raw_response": vlm_response
                }
                with open(debug_dirs["4_vlm_output"] / vlm_out_fname, "w") as f:
                    json.dump(debug_data, f, indent=2)

                matched_event = None
                fallback_event = None
                for plate_value in plate_text:
                    event_payload = {
                        "time_sec": best_sample["timestamp"],
                        "frame": best_sample["frame_idx"],
                        "type": "vehicle_detected",
                        "description": f"Vehicle {data['class']} ID:{tid}",
                        "plate_text": plate_value,
                        "vlm_verified": True,
                        "vlm_response": vlm_response,
                        "thumbnail": evidence_thumbnail,
                        "bbox": vehicle_bbox,
                    }
                    if fallback_event is None:
                        fallback_event = event_payload

                    if self._plate_matches_target(plate_value, target_plate_hint):
                        matched_event = event_payload
                        full_fr = best_sample.get("full_frame")
                        save_frame = full_fr if full_fr is not None else ocr_crop
                        _save_matched_frame(
                            save_frame,
                            tid,
                            int(best_sample.get("frame_idx", -1)),
                            float(best_sample.get("timestamp", -1.0)),
                            plate_value,
                        )
                        break

                if target_plate_hint:
                    if matched_event:
                        final_events.append(matched_event)
                else:
                    chosen_event = matched_event or fallback_event
                    if chosen_event:
                        _save_matched_frame(
                            ocr_crop,
                            tid,
                            int(best_sample.get("frame_idx", -1)),
                            float(best_sample.get("timestamp", -1.0)),
                            chosen_event.get("plate_text"),
                        )
                        final_events.append(chosen_event)

            if _is_cancel_requested(cancel_check):
                break

        last_frame = frame_idx - 1
        logger.info("LPR scan complete: events=%s last_frame=%s", len(final_events), last_frame)
        if return_state:
            return final_events, last_frame
        return final_events
    
class CrowdCountSkill:
    """
    Refactored from crowd_counting.py
    Uses P2PNet (via imported model) to count people and generate density stats.
    """
    def __init__(self, weights_path="weights/SHTechA.pth"):
        logger.info("Initializing CrowdCountSkill...")
        self.weights_path = weights_path

    def process_video(self, video_path, **kwargs):
        logger.info("CrowdCount stub invoked: %s", video_path)
        return [{"time_sec": 0, "count": 0, "note": "Crowd module stub - requires external P2PNet deps"}]

class VLM_Forensic:
    """
    Forensic VLM that sends specific queries to the backend.
    """
    def __init__(self):
        from detectors import VLM_Lite
        self.base_vlm = VLM_Lite()

    def _normalize_frames(self, frames, max_frames):
        if not frames:
            return []
        cleaned = []
        for f in frames:
            if isinstance(f, (tuple, list)) and len(f) >= 2:
                cleaned.append(f[1])
            else:
                cleaned.append(f)
        if max_frames and len(cleaned) > max_frames:
            step = len(cleaned) / float(max_frames)
            cleaned = [cleaned[int(i * step)] for i in range(max_frames)]
        return cleaned

    def _build_montage(self, frames, cols):
        n = len(frames)
        cols = max(1, min(cols, n))
        rows = (n + cols - 1) // cols

        target_w = getattr(config, "VLM_RESIZE_WIDTH", 640)
        target_h = getattr(config, "VLM_RESIZE_HEIGHT", 480)
        cell_w = max(1, target_w // cols)
        cell_h = max(1, target_h // rows)

        montage = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)

        for idx, frame in enumerate(frames):
            r = idx // cols
            c = idx % cols
            resized = cv2.resize(frame, (cell_w, cell_h))
            y1 = r * cell_h
            y2 = y1 + cell_h
            x1 = c * cell_w
            x2 = x1 + cell_w
            montage[y1:y2, x1:x2] = resized

        return montage

    def ask_sequence(self, frames, query, max_frames=6, cols=3):
        frames = self._normalize_frames(frames, max_frames)
        if not frames:
            logger.warning("VLM sequence called with no frames")
            return {"found": False, "error": "no_frames"}
        if len(frames) == 1:
            return self.ask(frames[0], query)

        montage = self._build_montage(frames, cols)
        return self.ask(montage, f"In any of these frames, is there {query}")

    def ask(self, frame_array, query, timeout_sec=60, max_retries=2):
        import time
        start = time.time()
        logger.debug("VLM ask start: query=%s", query)
        try:
            # UPDATED: Don't resize, maintain original quality for VLM
            # frame_resized = cv2.resize(frame_array, (config.VLM_RESIZE_WIDTH, config.VLM_RESIZE_HEIGHT))
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            ret, buffer = cv2.imencode('.jpg', frame_array, encode_params)
            if not ret:
                logger.error("VLM frame encoding failed")
                return {"error": "encoding_fail"}

            prompt_text = (
                f"Analyze this image. Question: {query}? "
                "Return JSON: {'answer': 'string', 'found': boolean, 'confidence': float}"
            )

            files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}

            retries = max(1, int(max_retries))
            timeout = max(1, int(timeout_sec))
            for attempt in range(retries):
                try:
                    response = self.base_vlm.session.post(
                        self.base_vlm.endpoint,
                        files=files,
                        data={"prompt": prompt_text},
                        timeout=timeout,
                    )
                    break
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
                    logger.warning(f"VLM Request failed (attempt {attempt + 1}/{retries}): {e}")
                    if attempt < retries - 1:
                        time.sleep(2 * (attempt + 1))
                    else:
                        raise e

            if response.status_code == 200:
                try:
                    res_json = response.json()
                    logger.debug("VLM response OK in %.2fs", time.time() - start)
                    if "choices" in res_json:
                        content = res_json["choices"][0]["message"]["content"]
                        if "```" in content:
                            content = content.replace("```json", "").replace("```", "")
                        return json.loads(content)
                    return res_json
                except Exception:
                    return {"found": False, "raw": response.text}
            logger.warning("VLM non-200 response: %s", response.status_code)
            return {"found": False, "error": response.status_code}

        except Exception as e:
            logger.exception("VLM ask failed: %s", e)
            return {"found": False, "error": str(e)}

class GeneralEventSkill:
    """
    Uses BytePlus Seed Video Query to scan for semantic events.
    Replaces frame-by-frame approach with direct video query.
    """
    def __init__(self):
        import os
        self.api_key = os.getenv("SEED_API_KEY", "")
        self.base_url = os.getenv("SEED_BASE_URL", "https://ark.ap-southeast.bytepluses.com/api/v3")
        self.endpoint_id = "ep-20260210203114-llmhr"
        self.seed_query_module = "seed_video_query"
        # Compression settings
        self.compress_before_upload = True
        self.compress_fps = 3
        self.compress_max_width = 512
        self.compress_crf = 25
        self.compress_preset = "veryfast"
        self.compress_output_dir = "./compressed_storage"
        self.fps = 2

    def _get_cancel_check(self, kwargs):
        """Get cancel check function from kwargs or config"""
        cancel_check = kwargs.get("cancel_check")
        if callable(cancel_check):
            return cancel_check
        
        # Try to get from config
        try:
            from detectors_forensic import _is_cancel_requested
            return lambda: _is_cancel_requested(cancel_check)
        except ImportError:
            return None

    def _load_seed_query_video(self):
        """Load the Seed fast-query entry point used by the GENERAL tool."""
        import importlib

        try:
            module = importlib.import_module(self.seed_query_module)
        except Exception as exc:
            logger.error("Failed to import %s: %s", self.seed_query_module, exc)
            return None

        query_video = getattr(module, "query_video", None)
        if not callable(query_video):
            logger.error("%s.query_video is not callable", self.seed_query_module)
            return None
        return query_video

    def process_video(
        self,
        video_path,
        query,
        context_sec=2.0,
        start_frame=0,
        end_frame=None,
        show_progress=True,
        return_state=False,
        frame_stride=None,
        reference_image=None,
        timeout=300,
        **kwargs,
    ):
        """        
        Args:
            video_path: Path to video file
            query: Text query for the video
            context_sec: Context seconds (for future use)
            start_frame: Start frame index
            end_frame: End frame index
            show_progress: Show progress bar
            return_state: Return last frame info
            frame_stride: Frame stride (passed for compatibility)
            reference_image: Optional reference image
            timeout: Request timeout in seconds
            **kwargs: Additional args including cancel_check
        
        Returns:
            List of events or (events, last_frame) if return_state=True
        """
        import asyncio
        from pathlib import Path
        
        # Get cancel check
        cancel_check = self._get_cancel_check(kwargs)
        
        # Import seed query function
        seed_query_video = self._load_seed_query_video()
        if seed_query_video is None:
            return ([], int(start_frame or 0)) if return_state else []
        
        logger.info("GeneralEvent Seed query: path=%s query=%s", video_path, query)
        
        # Validate video
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            logger.error("Video not found: %s", video_path)
            return ([], int(start_frame or 0)) if return_state else []
        
        # Get video info
        cap = VideoReader(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video: %s", video_path)
            cap.release()
            return ([], int(start_frame or 0)) if return_state else []
        
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = getattr(config, "FPS", 30)
        
        start_frame = int(start_frame or 0)
        if end_frame is None:
            end_frame = total_frames - 1
        
        cap.release()
        
        logger.info("Video info: frames=%s fps=%s start=%s end=%s", 
                    total_frames, fps, start_frame, end_frame)
        
        # Run seed query
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                seed_query_video(
                    video_path=str(video_path_obj),
                    query=query,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.endpoint_id,
                    fps=self.fps,
                    reference_image=reference_image,
                    timeout=timeout,
                    cancel_check=cancel_check,
                    compress_before_upload=self.compress_before_upload,
                    compress_fps=self.compress_fps,
                    compress_max_width=self.compress_max_width,
                    compress_crf=self.compress_crf,
                    compress_preset=self.compress_preset,
                    compress_output_dir=self.compress_output_dir,
                    keep_compressed=True,
                )
            )
            
            # Extract events from result
            if not isinstance(result, dict):
                logger.error("Seed query returned unexpected result type: %s", type(result).__name__)
                return ([], int(start_frame or 0)) if return_state else []

            events = []
            raw_events = result.get("events", [])
            for evt in raw_events:
                # Ensure we capture bbox if provided by VLM
                bbox = evt.get("bbox")
                if bbox and isinstance(bbox, list) and len(bbox) == 4:
                    # Convert normalized [ymin, xmin, ymax, xmax] (0-1000) to pixel [x1, y1, x2, y2]
                    try:
                        ymin, xmin, ymax, xmax = bbox
                        x1 = int(xmin * width / 1000.0)
                        y1 = int(ymin * height / 1000.0)
                        x2 = int(xmax * width / 1000.0)
                        y2 = int(ymax * height / 1000.0)
                        bbox = [x1, y1, x2, y2]
                    except Exception as e:
                        logger.warning("BBox conversion failed: %s", e)
                        bbox = None

                events.append({
                    "time_sec": evt.get("time_sec", evt.get("timestamp", 0)),
                    "description": evt.get("description", "VLM Event"),
                    "bbox": bbox,
                })

            summary = result.get("summary", {})
            
            logger.info("Seed query complete: events=%s status=%s", 
                       len(events), summary.get("status", "unknown"))
            
            # Log timing info
            timing = summary.get("timing", {})
            if timing:
                logger.info("Timing: compression=%.1fs upload=%.1fs processing=%.1fs request=%.1fs total=%.1fs",
                           timing.get("compression_time", 0),
                           timing.get("upload_time", 0),
                           timing.get("processing_wait_time", 0),
                           timing.get("request_time", 0),
                           timing.get("total_time", 0))
            
            if return_state:
                return events, end_frame
            return events
            
        except Exception as e:
            logger.error("Seed query failed: %s", e)
            import traceback
            traceback.print_exc()
            return ([], int(start_frame or 0)) if return_state else []
        finally:
            if loop is not None:
                asyncio.set_event_loop(None)
                loop.close()

class WeaponDetectionSkill:
    """
    Detects weapons (guns, knives) using YOLO-World.
    Implements temporal persistence to remove false positives.
    """
    def __init__(self, model_path="yolov8s-world.pt"):
        logger.info("Initializing WeaponDetectionSkill...")
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        try:
            self.model = YOLO(model_path)
            if "world" in model_path:
                self.model.set_classes(["gun", "pistol", "rifle", "knife"])
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not load WeaponDetectionSkill model: {e}")
            self.model = None

    def process_video(
        self,
        video_path,
        start_frame=0,
        end_frame=None,
        show_progress=True,
        return_state=False,
        **kwargs,
    ):
        if self.model is None:
            logger.error("Weapon model not loaded")
            return ([], int(start_frame or 0)) if return_state else []
        # Favor preprocessed frames for detection phase if available
        input_source = kwargs.get("preprocessed_dir") or video_path
        cap = VideoReader(input_source)
        if not cap.isOpened():
            logger.error("WeaponDetectionSkill failed to open video source: %s", input_source)
            return ([], int(start_frame or 0)) if return_state else []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        logger.info("Weapon scan start: path=%s fps=%s start=%s end=%s", video_path, fps, start_frame, end_frame)

        events = []
        active_counts = {}
        CONF_THRESH = 0.5
        TIME_THRESH_SEC = 1.0
        FRAMES_THRESH = int(fps * TIME_THRESH_SEC)

        frame_idx = int(start_frame or 0)
        cancel_check = kwargs.get("cancel_check")
        pbar = tqdm(desc="Scanning for Weapons", disable=not show_progress or end_frame is not None)

        while cap.isOpened():
            if _is_cancel_requested(cancel_check):
                logger.info("Weapon scan canceled")
                break
            if end_frame is not None and frame_idx > end_frame:
                break
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pbar.update(1)

            results = self.model(frame, verbose=False, conf=CONF_THRESH)

            current_detected = set()
            best_bbox_by_label = {}
            best_conf_by_label = {}

            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = (
                    results[0].boxes.conf.cpu().numpy()
                    if getattr(results[0].boxes, "conf", None) is not None
                    else None
                )
                for i, cls_id in enumerate(classes):
                    label = self.model.names[cls_id]
                    if label not in ["gun", "pistol", "rifle", "knife"]:
                        continue
                    current_detected.add(label)
                    conf = float(confs[i]) if confs is not None else 0.0
                    prev_conf = best_conf_by_label.get(label, -1.0)
                    if conf >= prev_conf:
                        x1, y1, x2, y2 = map(int, boxes[i])
                        best_conf_by_label[label] = conf
                        best_bbox_by_label[label] = [x1, y1, x2, y2]

            for label in current_detected:
                active_counts[label] = active_counts.get(label, 0) + 1

            for label in list(active_counts.keys()):
                if label not in current_detected:
                    active_counts[label] = 0

            for label, count in active_counts.items():
                if count == FRAMES_THRESH:
                    timestamp = frame_idx / fps
                    desc = f"Confirmed {label} visible for {TIME_THRESH_SEC}s"
                    logger.warning("WEAPON ALERT: %s", desc)
                    events.append({
                        "time_sec": timestamp,
                        "type": "weapon_detected",
                        "description": desc,
                        "object": label,
                        "thumbnail": _encode_thumbnail(frame),
                        "bbox": best_bbox_by_label.get(label),
                    })

        cap.release()
        last_frame = frame_idx - 1
        logger.info("Weapon scan complete: events=%s last_frame=%s", len(events), last_frame)
        if return_state:
            return events, last_frame
        return events

class VehicleColorSkill:
    """
    Detects vehicles and filters by dominant color described in the query
    (e.g. "red car", "blue truck").
    """
    COLOR_RANGES = {
        "red": [((0, 50, 50), (10, 255, 255)), ((170, 50, 50), (180, 255, 255))],
        "blue": [((100, 50, 50), (130, 255, 255))],
        "green": [((40, 50, 50), (85, 255, 255))],
        "yellow": [((20, 70, 70), (35, 255, 255))],
        "orange": [((10, 70, 70), (20, 255, 255))],
        "white": [((0, 0, 200), (180, 40, 255))],
        "black": [((0, 0, 0), (180, 255, 40))],
        "gray": [((0, 0, 40), (180, 30, 200))],
        "grey": [((0, 0, 40), (180, 30, 200))],
        "silver": [((0, 0, 80), (180, 30, 220))],
    }

    VEHICLE_KEYWORDS = {
        "car": ["car", "sedan", "hatchback", "suv", "van"],
        "truck": ["truck", "lorry", "pickup"],
        "bus": ["bus", "coach"],
        "motorcycle": ["motorcycle", "motorbike", "bike"],
    }
    SCAN_FPS = 4.0
    MIN_AREA = 2500
    MIN_RATIO = 0.07
    MIN_GAP_SEC = 0.5
    TRACK_REUSE_GAP_SEC = 4.0
    DET_CONF_MIN = 0.20
    RED_LIGHT_SMALL_BLOB_RATIO = 0.01
    RED_LIGHT_BRIGHT_V = 225

    def __init__(self, model_path="yolov8m.pt"):
        logger.info("Initializing VehicleColorSkill...")
        self.device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not load VehicleColorSkill model: {e}")
            self.model = None

        self.class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def _parse_query(self, query):
        q = (query or "").lower()
        color = None
        for name in self.COLOR_RANGES.keys():
            if name in q:
                color = "gray" if name == "grey" else name
                break

        vehicle = None
        for v_type, keywords in self.VEHICLE_KEYWORDS.items():
            if any(k in q for k in keywords):
                vehicle = v_type
                break

        logger.debug("VehicleColor parsed query: color=%s vehicle=%s", color, vehicle)
        return color, vehicle

    def _color_ratio(self, crop, color):
        if crop is None or crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ranges = self.COLOR_RANGES.get(color, [])
        if not ranges:
            return 0.0
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask_total = cv2.bitwise_or(mask_total, cv2.inRange(hsv, lower_np, upper_np))

        # Red vehicle queries are vulnerable to tail/brake lights.
        # Remove tiny, very bright red blobs before computing the ratio.
        if color == "red":
            mask_total = self._suppress_red_light_spots(mask_total, hsv)

        return float(np.count_nonzero(mask_total)) / float(mask_total.size)

    def _suppress_red_light_spots(self, red_mask, hsv):
        if red_mask is None or red_mask.size == 0:
            return red_mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        if num_labels <= 1:
            return red_mask

        value_channel = hsv[:, :, 2]
        total_px = float(red_mask.size)
        cleaned = red_mask.copy()

        # Tune for tiny bright hotspots (typical tail/brake lights).
        # Keep this conservative so we don't erase true red body paint.
        small_blob_ratio = self.RED_LIGHT_SMALL_BLOB_RATIO
        bright_v_thresh = self.RED_LIGHT_BRIGHT_V

        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            blob_ratio = area / total_px
            if blob_ratio > small_blob_ratio:
                continue

            component_mask = (labels == lbl)
            mean_v = float(value_channel[component_mask].mean()) if np.any(component_mask) else 0.0
            if mean_v >= bright_v_thresh:
                cleaned[component_mask] = 0

        return cleaned

    def process_video(
        self,
        video_path,
        query,
        start_frame=0,
        end_frame=None,
        show_progress=True,
        return_state=False,
        **kwargs,
    ):
        if self.model is None:
            logger.error("VehicleColor model not loaded")
            return ([], int(start_frame or 0)) if return_state else []

        color, vehicle = self._parse_query(query)
        if not color:
            logger.info("VehicleColor: no color found in query: %s", query)
            return ([], int(start_frame or 0)) if return_state else []

        # Favor preprocessed frames for detection phase if available
        input_source = kwargs.get("preprocessed_dir") or video_path
        cap = VideoReader(input_source)
        if not cap.isOpened():
            logger.error("VehicleColorSkill failed to open video source: %s", input_source)
            return ([], int(start_frame or 0)) if return_state else []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = getattr(config, "FPS", 30)
        logger.info("VehicleColor scan start: color=%s vehicle=%s path=%s fps=%s", color, vehicle, video_path, fps)

        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segment_total = (
            (end_frame - start_frame + 1)
            if end_frame is not None
            else max(0, total_frames - start_frame)
        )

        scan_fps = self.SCAN_FPS
        frame_stride = max(1, int(round(fps / scan_fps)))
        min_area = self.MIN_AREA
        min_ratio = self.MIN_RATIO
        min_gap = self.MIN_GAP_SEC
        track_reuse_gap_sec = self.TRACK_REUSE_GAP_SEC
        det_conf_min = self.DET_CONF_MIN
        results = []
        fallback_results = []
        last_emit_time = -1.0
        last_fallback_time = -1.0
        emitted_track_times = {}
        fallback_emitted_track_times = {}
        frame_idx = int(start_frame or 0)
        cancel_check = kwargs.get("cancel_check")
        logger.info(
            "VehicleColor thresholds: scan_fps=%s min_area=%s min_ratio=%s min_gap=%s reuse_gap=%s det_conf=%s",
            scan_fps, min_area, min_ratio, min_gap, track_reuse_gap_sec, det_conf_min
        )

        pbar = tqdm(
            total=segment_total,
            desc=f"Color scan '{color}'",
            disable=not show_progress or end_frame is not None,
        )

        while cap.isOpened():
            if _is_cancel_requested(cancel_check):
                logger.info("VehicleColor scan canceled")
                break
            if end_frame is not None and frame_idx > end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            current_idx = frame_idx
            frame_idx += 1
            pbar.update(1)

            if torch:
                with torch.inference_mode():
                    det_results = self.model.track(
                        frame,
                        persist=True,
                        verbose=False,
                        classes=[2, 3, 5, 7],
                        tracker="botsort.yaml",
                        conf=det_conf_min,
                    )
            else:
                det_results = self.model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    classes=[2, 3, 5, 7],
                    tracker="botsort.yaml",
                    conf=det_conf_min,
                )

            if not det_results or det_results[0].boxes is None:
                continue

            boxes = det_results[0].boxes.xyxy.cpu().numpy()
            classes = det_results[0].boxes.cls.cpu().numpy().astype(int)
            ids_tensor = getattr(det_results[0].boxes, "id", None)
            if ids_tensor is not None:
                track_ids = ids_tensor.cpu().numpy().astype(int).tolist()
            else:
                track_ids = [None] * len(classes)
            confs = (
                det_results[0].boxes.conf.cpu().numpy()
                if getattr(det_results[0].boxes, "conf", None) is not None
                else None
            )

            for i, cls_id in enumerate(classes):
                if _is_cancel_requested(cancel_check):
                    logger.info("VehicleColor scan canceled")
                    break
                label = self.class_names.get(cls_id)
                track_id = track_ids[i] if i < len(track_ids) else None

                x1, y1, x2, y2 = map(int, boxes[i])
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue

                cx1 = int(x1 + (x2 - x1) * 0.2)
                cy1 = int(y1 + (y2 - y1) * 0.2)
                cx2 = int(x2 - (x2 - x1) * 0.2)
                cy2 = int(y2 - (y2 - y1) * 0.2)
                crop = frame[cy1:cy2, cx1:cx2]

                ratio = self._color_ratio(crop, color)
                if ratio < min_ratio:
                    continue

                time_sec = current_idx / fps
                det_conf = float(confs[i]) if confs is not None else None
                event = {
                    "time_sec": time_sec,
                    "frame": current_idx,
                    "type": "vehicle_color",
                    "description": f"{color} {label or 'vehicle'}",
                    "color": color,
                    "vehicle": label,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "det_conf": det_conf,
                    "vehicle_match": (label == vehicle) if vehicle else True,
                    "thumbnail": _encode_thumbnail(frame),
                }

                if vehicle and label != vehicle:
                    if track_id is not None:
                        prev_time = fallback_emitted_track_times.get(track_id)
                        if prev_time is not None and (time_sec - prev_time) < track_reuse_gap_sec:
                            continue
                        fallback_emitted_track_times[track_id] = time_sec
                    else:
                        if last_fallback_time >= 0 and (time_sec - last_fallback_time) < min_gap:
                            continue
                        last_fallback_time = time_sec
                    fallback_results.append(event)
                else:
                    if track_id is not None:
                        prev_time = emitted_track_times.get(track_id)
                        if prev_time is not None and (time_sec - prev_time) < track_reuse_gap_sec:
                            continue
                        emitted_track_times[track_id] = time_sec
                    else:
                        if last_emit_time >= 0 and (time_sec - last_emit_time) < min_gap:
                            continue
                        last_emit_time = time_sec
                    results.append(event)

        cap.release()
        last_frame = frame_idx - 1
        logger.info("VehicleColor scan complete: results=%s last_frame=%s", len(results), last_frame)
        if vehicle and not results and fallback_results:
            results = fallback_results
        if return_state:
            return results, last_frame
        return results
