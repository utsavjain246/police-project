"""
BytePlus Seed Video Query Module

Usage:
    from seed_fast_query import query_video
    
    # Simple usage
    result = await query_video("video.mp4", "Find all persons in red dress")
    
    # With cancel support
    cancel_flag = False
    result = await query_video("video.mp4", "query", cancel_check=lambda: cancel_flag)
"""
import asyncio
import logging
import re
import sys
from pathlib import Path
import time
import subprocess
import base64
from config import SEED_API_KEY, SEED_BASE_URL
from typing import Optional, Callable

from byteplussdkarkruntime import AsyncArk
import hashlib

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "video_path": "C:/Users/91701/Downloads/Project-ui/backend/clips/clip_002.mp4",
    "query": "Find all the persons in red dress and return their exact timestamps.",
    "api_key": SEED_API_KEY,
    "base_url": SEED_BASE_URL,
    "model": "ep-20260210203114-llmhr",
    "fps": 2,
    "reference_image": None,
    "timeout": 300,
    "compress_before_upload": True,
    "compress_fps": 3,
    "compress_max_width": 512,
    "compress_crf": 25,
    "compress_preset": "veryfast",
    "compress_output_dir": "./compressed_storage",
}
# ============================================================

logger = logging.getLogger("seed_query")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s"
)


# ============================================================
# Helper Functions
# ============================================================

def encode_image(image_path: Path) -> str:
    """Encode image to base64 data URL"""
    ext = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    with image_path.open("rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def encode_frame_to_base64(frame_data: bytes) -> str:
    """Encode raw frame bytes to base64 data URL"""
    b64 = base64.b64encode(frame_data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def compress_video(input_path: Path, output_path: Path, fps: int, max_width: int, crf: int, preset: str) -> None:
    """Compress video for upload using ffmpeg"""
    vf = f"fps={fps}"
    if max_width and max_width > 0:
        vf = f"scale='min({max_width},iw)':-2,{vf}"

    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-an", "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _get_compressed_path(input_path: Path, output_dir: Path, fps: int, max_width: int, crf: int, preset: str) -> Path:
    """Generate a deterministic compressed path based on compression parameters and file metadata"""
    # Include file size and mtime to ensure hash stays same even if file is renamed (within same machine/mod time)
    # or changes content but keeps name.
    try:
        stat = input_path.stat()
        file_meta = f"{stat.st_size}_{int(stat.st_mtime)}"
    except Exception:
        file_meta = input_path.name

    settings_str = f"{file_meta}_{fps}_{max_width}_{crf}_{preset}"
    settings_hash = hashlib.md5(settings_str.encode()).hexdigest()[:12]
    # Solely hash-based filename for reuse even across renames
    return output_dir / f"vid_{settings_hash}.mp4"


def _is_cancel_requested(cancel_check: Optional[Callable[[], bool]]) -> bool:
    """Check if cancellation was requested"""
    if not callable(cancel_check):
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _coerce_time_sec(value, video_fps: float = 30.0) -> Optional[float]:
    """Convert timestamp-like values into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    frame_match = re.search(r"frame\s+(?:number\s+)?(\d+)", text, re.IGNORECASE)
    if frame_match:
        frame_num = int(frame_match.group(1))
        return frame_num / video_fps if video_fps > 0 else 0.0

    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 1:
                return float(parts[0])
        except ValueError:
            return None

    numeric_match = re.search(r"(\d+(?:\.\d+)?)", text)
    if numeric_match:
        return float(numeric_match.group(1))

    return None


def _parse_timestamps(response_text: str, video_fps: float = 30.0) -> list:
    """
    Parse timestamps from LLM response text.
    Supports various formats:
    - "at 00:01:23.456" / "at 1:23"
    - "timestamp: 1.5s" / "timestamp: 1.5 seconds"
    - "frame 45" / "frame number 45"
    - "1.5 seconds into the video"
    - JSON: {"timestamps": [...]} or {"events": [...]}
    """
    events = []
    
    # Try to extract JSON first
    try:
        import json
        json_match = re.search(r'\[.*\]|\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            ts = item.get('timestamp') or item.get('time') or item.get('time_sec') or item.get('t')
                            time_sec = _coerce_time_sec(ts, video_fps)
                            if time_sec is not None:
                                events.append({
                                    'time_sec': time_sec,
                                    'description': item.get('description') or item.get('desc') or item.get('text') or '',
                                    'confidence': item.get('confidence') or item.get('conf') or 'HIGH',
                                })
                elif isinstance(data, dict):
                    timestamps = data.get('timestamps') or data.get('times') or data.get('events') or []
                    if isinstance(timestamps, list):
                        for ts in timestamps:
                            if isinstance(ts, dict):
                                time_sec = _coerce_time_sec(
                                    ts.get('timestamp') or ts.get('time') or ts.get('time_sec') or ts.get('t'),
                                    video_fps,
                                )
                                if time_sec is None:
                                    continue
                                events.append({
                                    'time_sec': time_sec,
                                    'description': ts.get('description') or ts.get('desc') or ts.get('text') or '',
                                    'confidence': ts.get('confidence') or ts.get('conf') or 'HIGH',
                                    'bbox': ts.get('bbox'),
                                })
                            else:
                                time_sec = _coerce_time_sec(ts, video_fps)
                                if time_sec is not None:
                                    events.append({'time_sec': time_sec, 'description': '', 'confidence': 'HIGH'})
                    elif data.get('timestamp') or data.get('time'):
                        time_sec = _coerce_time_sec(data.get('timestamp') or data.get('time'), video_fps)
                        if time_sec is not None:
                            events.append({
                                'time_sec': time_sec,
                                'description': data.get('description') or '',
                                'confidence': data.get('confidence') or 'HIGH',
                            })
                if events:
                    unique_events = {}
                    for ev in events:
                        key = (round(ev.get('time_sec', 0), 2), ev.get('frame'))
                        if key not in unique_events:
                            unique_events[key] = ev
                    return sorted(unique_events.values(), key=lambda x: x.get('time_sec', 0))
            except json.JSONDecodeError:
                pass
    except Exception:
        pass
    
    # Parse text patterns
    patterns = [
        (r'(?:at\s+|timestamp[:\s]+|time[:\s]+)(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)\s*(?:seconds?|s)?', 'time_str'),
        (r'(\d+(?:\.\d+)?)\s*(?:seconds?|s)\s*(?:into\s+the\s+video)?', 'seconds'),
        (r'frame\s+(?:number\s+)?(\d+)', 'frame'),
    ]
    
    for pattern, ptype in patterns:
        matches = re.finditer(pattern, response_text, re.IGNORECASE)
        for match in matches:
            try:
                if ptype == 'time_str':
                    time_str = match.group(1)
                    parts = time_str.split(':')
                    if len(parts) == 3:
                        seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                    elif len(parts) == 2:
                        seconds = int(parts[0]) * 60 + float(parts[1])
                    else:
                        seconds = float(parts[0])
                    events.append({'time_sec': seconds, 'description': '', 'confidence': 'HIGH'})
                elif ptype == 'seconds':
                    seconds = float(match.group(1))
                    events.append({'time_sec': seconds, 'description': '', 'confidence': 'HIGH'})
                elif ptype == 'frame':
                    frame_num = int(match.group(1))
                    seconds = frame_num / video_fps if video_fps > 0 else 0
                    events.append({'time_sec': seconds, 'description': '', 'confidence': 'HIGH', 'frame': frame_num})
            except (ValueError, IndexError):
                continue
    
    # Remove duplicates and sort
    unique_events = {}
    for ev in events:
        key = round(ev.get('time_sec', 0), 2)
        if key not in unique_events:
            unique_events[key] = ev
    
    return sorted(unique_events.values(), key=lambda x: x.get('time_sec', 0))


def _parse_confidence(conf_str: str) -> str:
    """Normalize confidence string"""
    if isinstance(conf_str, (int, float)):
        if conf_str >= 0.8:
            return 'HIGH'
        elif conf_str >= 0.5:
            return 'MEDIUM'
        return 'LOW'
    
    conf_lower = str(conf_str).upper()
    if 'HIGH' in conf_lower or 'CERTAIN' in conf_lower:
        return 'HIGH'
    elif 'MEDIUM' in conf_lower or 'MODERATE' in conf_lower:
        return 'MEDIUM'
    return 'LOW'


# ============================================================
# Main Query Function
# ============================================================

async def query_video(
    video_path: str,
    query: str,
    api_key: str = None,
    base_url: str = None,
    model: str = None,
    fps: int = 2,
    reference_image: str = None,
    timeout: int = 300,
    cancel_check: Optional[Callable[[], bool]] = None,
    compress_before_upload: bool = True,
    compress_fps: int = 3,
    compress_max_width: int = 512,
    compress_crf: int = 25,
    compress_preset: str = "veryfast",
    compress_output_dir: str = "./compressed_storage",
    keep_compressed: bool = False,
) -> dict:
    """
    Query a video with text using the BytePlus Seed model.
    
    Returns a frontend-compatible result dictionary:
    {
        "events": [
            {
                "time_sec": float,
                "type": str,
                "description": str,
                "confidence": "HIGH/MEDIUM/LOW",
                "thumbnail": str (optional),
                "bbox": [x1, y1, x2, y2] (optional),
                "frame": int (optional),
                "raw_response": str,
            }
        ],
        "summary": {
            "total_events": int,
            "video_path": str,
            "query": str,
            "timing": {...}
        }
    }
    """
    # Use defaults from CONFIG if not provided
    if api_key is None:
        api_key = CONFIG["api_key"]
    if base_url is None:
        base_url = CONFIG["base_url"]
    if model is None:
        model = CONFIG["model"]
    
    logger = logging.getLogger("seed_query")
    
    video_path_obj = Path(video_path).expanduser().resolve()
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video not found: {video_path_obj}")
    
    logger.info(f"Processing video: {video_path_obj}")
    logger.info(f"Query: {query}")
    
    # Timing tracking
    timing = {
        "compression_time": 0.0,
        "upload_time": 0.0,
        "processing_wait_time": 0.0,
        "request_time": 0.0,
        "total_time": 0.0,
    }
    
    total_start = time.perf_counter()
    client = AsyncArk(base_url=base_url, api_key=api_key)
    upload_path = video_path_obj
    compressed_path = None
    file_id = None
    
    try:
        # Step 1: Compress video if needed
        if compress_before_upload:
            output_dir = Path(compress_output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            compressed_path = _get_compressed_path(
                video_path_obj, 
                output_dir, 
                compress_fps, 
                compress_max_width, 
                compress_crf, 
                compress_preset
            )
            
            if compressed_path.exists():
                logger.info(f"Using existing compressed video: {compressed_path}")
                upload_path = compressed_path
            else:
                logger.info(f"Compressing video before upload to: {compressed_path}")
                try:
                    comp_start = time.perf_counter()
                    await asyncio.to_thread(
                        compress_video,
                        video_path_obj,
                        compressed_path,
                        compress_fps,
                        compress_max_width,
                        compress_crf,
                        compress_preset,
                    )
                    timing["compression_time"] = time.perf_counter() - comp_start
                    upload_path = compressed_path
                    logger.info(f"Compression complete: {timing['compression_time']:.1f}s")
                except Exception as e:
                    logger.warning(f"Compression failed, falling back to original: {e}")
                    compressed_path = None
                    upload_path = video_path_obj
        
        # Check cancellation after compression
        if _is_cancel_requested(cancel_check):
            return _empty_result(video_path, query, "cancelled", timing)
        
        # Step 2: Upload video
        logger.info("Uploading video...")
        upload_start = time.perf_counter()
        
        if _is_cancel_requested(cancel_check):
            return _empty_result(video_path, query, "cancelled", timing)
        
        with open(upload_path, "rb") as video_file:
            file_obj = await client.files.create(
                file=video_file,
                purpose="user_data",
                preprocess_configs={"video": {"fps": fps}},
            )
        
        file_id = file_obj.id
        timing["upload_time"] = time.perf_counter() - upload_start
        logger.info(f"Video uploaded (id={file_id}, time={timing['upload_time']:.1f}s)")
        
        if _is_cancel_requested(cancel_check):
            try:
                await client.files.delete(file_id)
            except:
                pass
            return _empty_result(video_path, query, "cancelled", timing)
        
        # Step 3: Wait for processing
        logger.info("Waiting for video processing...")
        processing_start = time.perf_counter()
        
        await client.files.wait_for_processing(file_id, poll_interval=2.0)
        timing["processing_wait_time"] = time.perf_counter() - processing_start
        logger.info("Video processed")
        
        if _is_cancel_requested(cancel_check):
            return _empty_result(video_path, query, "cancelled", timing)
        
        # Step 4: Build content
        content = [{"type": "input_video", "file_id": file_id}]
        
        if reference_image:
            ref_path = Path(reference_image).expanduser().resolve()
            if ref_path.exists():
                image_url = encode_image(ref_path)
                content.append({"type": "input_image", "image_url": image_url})
                logger.info(f"Added reference image: {reference_image}")
        
        # Updated prompt to request bboxes and structured JSON
        system_instructions = (
            "Analyze the video and identify EVERY occurrence of the events described in the user query. "
            "Do NOT summarize multiple occurrences into one. List each event separately with its own timestamp. "
            "For each event, provide a timestamp (seconds), a concise description, "
            "and a bounding box for the primary subject. "
            "Bounding boxes MUST be in the format [ymin, xmin, ymax, xmax] using normalized coordinates (0-1000). "
            "Return the results as a strict JSON object with an 'events' list."
        )
        full_query = f"USER QUERY: {query}\n\n{system_instructions}"
        content.append({"type": "input_text", "text": full_query})
        
        # Step 5: Send query
        logger.info("Sending query to model...")
        request_start = time.perf_counter()
        
        try:
            response = await asyncio.wait_for(
                client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": content}],
                ),
                timeout=timeout,
            )
            timing["request_time"] = time.perf_counter() - request_start
            logger.info(f"Response received in {timing['request_time']:.1f}s")
            
            # Extract output text
            output_text = ""
            if hasattr(response, 'output_text') and response.output_text:
                output_text = response.output_text
            elif hasattr(response, 'output') and response.output:
                output = response.output
                if hasattr(output, 'text'):
                    output_text = output.text
                elif isinstance(output, list):
                    for item in output:
                        if hasattr(item, 'content'):
                            for c in item.content:
                                if hasattr(c, 'text'):
                                    output_text = c.text
                                    break
                        if output_text:
                            break
            
            if _is_cancel_requested(cancel_check):
                return _build_result(video_path, query, output_text, timing, "cancelled")
            
            # Parse timestamps from response
            parsed_events = _parse_timestamps(output_text, fps)
            
            # Build events list
            events = []
            for ev in parsed_events:
                events.append({
                    "time_sec": ev.get('time_sec'),
                    "type": query,
                    "description": ev.get('description') or f"Detected: {query}",
                    "confidence": _parse_confidence(ev.get('confidence', 'HIGH')),
                    "thumbnail": None,
                    "bbox": ev.get('bbox'),
                    "frame": ev.get('frame'),
                    "raw_response": output_text,
                })
            
            # If no events parsed but we have a response, create a single event
            if not events and output_text:
                events.append({
                    "time_sec": 0,
                    "type": query,
                    "description": output_text[:500],
                    "confidence": "MEDIUM",
                    "thumbnail": None,
                    "bbox": None,
                    "frame": None,
                    "raw_response": output_text,
                })
            
            timing["total_time"] = time.perf_counter() - total_start
            
            return {
                "events": events,
                "summary": {
                    "total_events": len(events),
                    "video_path": str(video_path),
                    "query": query,
                    "timing": timing,
                    "status": "completed",
                }
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after {timeout}s")
            timing["total_time"] = time.perf_counter() - total_start
            return _build_result(video_path, query, f"Error: Request timed out after {timeout}s", timing, "timeout")
            
    finally:
        await client.close()
        if compressed_path and compressed_path.exists() and not keep_compressed:
            try:
                logger.debug(f"Removing temporary compressed video: {compressed_path}")
                compressed_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove temporary compressed video {compressed_path}: {e}")


def _empty_result(video_path: str, query: str, status: str, timing: dict) -> dict:
    """Build empty result for cancelled/error cases"""
    return {
        "events": [],
        "summary": {
            "total_events": 0,
            "video_path": str(video_path),
            "query": query,
            "timing": timing,
            "status": status,
        }
    }


def _build_result(video_path: str, query: str, raw_response: str, timing: dict, status: str = "completed") -> dict:
    """Build result from raw response"""
    return {
        "events": [],
        "summary": {
            "total_events": 0,
            "video_path": str(video_path),
            "query": query,
            "timing": timing,
            "status": status,
            "raw_response": raw_response,
        }
    }


async def main():
    """CLI entry point for testing"""
    try:
        result = await query_video(
            video_path=CONFIG["video_path"],
            query=CONFIG["query"],
            api_key=CONFIG["api_key"],
            base_url=CONFIG["base_url"],
            model=CONFIG["model"],
            fps=CONFIG["fps"],
            reference_image=CONFIG.get("reference_image"),
            timeout=CONFIG.get("timeout", 300),
        )
        
        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        print(f"Events found: {result['summary']['total_events']}")
        
        for i, event in enumerate(result['events']):
            print(f"\n--- Event {i+1} ---")
            print(f"Time: {event['time_sec']:.2f}s")
            print(f"Type: {event['type']}")
            print(f"Description: {event['description'][:100]}...")
            print(f"Confidence: {event['confidence']}")
        
        print("\n" + "="*60)
        print("TIMING:")
        print("="*60)
        timing = result['summary']['timing']
        print(f"Compression: {timing['compression_time']:.1f}s")
        print(f"Upload: {timing['upload_time']:.1f}s")
        print(f"Processing wait: {timing['processing_wait_time']:.1f}s")
        print(f"Request: {timing['request_time']:.1f}s")
        print(f"Total: {timing['total_time']:.1f}s")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

