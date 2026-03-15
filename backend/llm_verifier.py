import base64
import json
import logging
import io
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np
import cv2
import base64
import config

# Configure Logging
logger = logging.getLogger(__name__)

class LLMVerifier:
    def __init__(self):
        self.enabled = False
        self.gemini = None
        self.model_name = getattr(config, "GEMINI_MODEL", "gemini-1.5-flash-001")  # Use Flash for speed
        
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            api_key = getattr(config, "GEMINI_API_KEY", None)
            if not api_key or "YOUR_KEY" in api_key:
                logger.warning("LLMVerifier disabled: GEMINI_API_KEY not found in config.")
                return

            genai.configure(api_key=api_key)
            
            # Safety Settings: BLOCK_NONE is crucial for forensic tools 
            # (otherwise it blocks images of guns/knives/accidents)
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            self.generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=100,
                temperature=0.1, # Low temp for deterministic T/F
                response_mime_type="application/json",
            )
            
            self.gemini = genai.GenerativeModel(self.model_name)
            self.enabled = True
            logger.info(f"LLMVerifier initialized with {self.model_name}")

        except ImportError:
            logger.error("LLMVerifier disabled: google-generativeai package missing. pip install google-generativeai")
        except Exception as e:
            logger.error(f"LLMVerifier init failed: {e}")

    def _decode_image(self, b64_string):
        """Robustly decodes base64 string to PIL Image."""
        if not b64_string:
            return None
        try:
            # Handle data URI scheme if present
            if "base64," in b64_string:
                _, b64_string = b64_string.split("base64,", 1)
            
            image_data = base64.b64decode(b64_string)
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            logger.debug(f"Image decode failed: {e}")
            return None

    def _crop_with_padding(self, pil_img, bbox, padding_pct=0.15):
        """
        Crops the image to the bbox with context padding.
        Returns original image if bbox is invalid.
        """
        if not bbox or len(bbox) != 4:
            return pil_img
        
        try:
            w, h = pil_img.size
            x1, y1, x2, y2 = bbox
            
            # Calculate padding
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(box_w * padding_pct)
            pad_y = int(box_h * padding_pct)

            nx1 = max(0, int(x1 - pad_x))
            ny1 = max(0, int(y1 - pad_y))
            nx2 = min(w, int(x2 + pad_x))
            ny2 = min(h, int(y2 + pad_y))
            
            if nx2 <= nx1 or ny2 <= ny1:
                return pil_img

            return pil_img.crop((nx1, ny1, nx2, ny2))
        except Exception:
            return pil_img

    def _generate_prompt(self, query, event):
        """Generates a context-aware prompt based on the detection type."""
        
        event_type = event.get("type", "generic")
        desc = event.get("description", "")
        ocr_text = event.get("plate_text", "")
        label = event.get("object") or event.get("class") or query
        
        base_instruction = (
            "You are a False Positive Filter for video analytics. "
            "Analyze the image and strictly output JSON: {\"verified\": boolean, \"reason\": \"short string\"}. "
        )

        if "plate" in event_type or ocr_text:
            # License Plate Logic
            return (
                "You are a False Positive Filter for video analytics. "
                "Analyze the image and strictly output JSON: {\"verified\": boolean, \"reason\": \"short string\", \"corrected_text\": \"string or null\"}. "
                f"Does the image contain a license plate with text similar to '{ocr_text}'? "
                "Allow for minor OCR errors (e.g. 0 vs O, 8 vs B). "
                "If the OCR text is wrong but you can clearly read the correct plate text, set verified=true and provide the correct text in 'corrected_text'. "
                "If the text is correct, you can leave 'corrected_text' as null or provide the correct text. "
                "If the text is completely different, unreadable, or no plate is visible, verified=false."
            )
        
        elif "weapon" in event_type or "gun" in str(label).lower() or "knife" in str(label).lower():
            # Weapon Logic
            return (
                f"{base_instruction}"
                f"Is there a REAL {label} visible in this crop? "
                "Ignore toys, phones, or hands if no weapon is clearly visible. "
                "If it looks like a real threat, verified=true."
            )
            
        elif "color" in event_type:
            # Vehicle Color Logic
            return (
                f"{base_instruction}"
                f"Does the image show a {query}? "
                "Focus on the dominant color of the vehicle. "
                "verified=true if it matches the description."
            )

        else:
            # General / Semantic Logic
            return (
                f"{base_instruction}"
                f"Does the image match this description: '{query or desc}'? "
                "Be objective."
            )

    async def _verify_single_event(self, event, query):
        """
        Asynchronously formats the prompt and calls Gemini for a single event.
        Returns a tuple: (bool_keep_event, dict_event)
        """
        try:
            img_b64 = event.get("thumbnail")
            if not img_b64:
                return True, event

            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return True, event # Keep by default if we can't check

            # 2. Smart Crop (Focus attention)
            # We crop if a bbox exists, otherwise use full frame
            target_img = self._crop_with_padding(img, event.get("bbox"))
            
            # 3. Build Prompt
            prompt = self._generate_prompt(query, event)

            # 4. Call Gemini (Async)
            response = await self.gemini.generate_content_async(
                [prompt, target_img],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                request_options={"timeout": 15.0}
            )
            
            # 5. Parse JSON
            try:
                raw_text = response.text.strip()
            except ValueError:
                # Occurs if response was blocked or empty
                raw_text = ""
                
            if not raw_text:
                logger.debug("Verifier: empty response from Gemini, keeping event")
                return True, event
                
            if "```json" in raw_text:
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            elif "```" in raw_text:
                raw_text = raw_text.replace("```", "").strip()
                
            if not raw_text:
                logger.debug("Verifier: empty JSON after cleanup, keeping event")
                return True, event
            
            try:
                result_json = json.loads(raw_text)
                is_verified = result_json.get("verified", False)
                reason = result_json.get("reason", "No reason provided")

                if is_verified:
                    event["verification_note"] = f"LLM Confirmed: {reason}"
                    
                    corrected_text = result_json.get("corrected_text")
                    if corrected_text and isinstance(corrected_text, str) and corrected_text.strip():
                        new_text = corrected_text.strip().upper()
                        if event.get("plate_text") and new_text != event.get("plate_text"):
                            logger.info(f"LLM corrected plate text from '{event['plate_text']}' to '{new_text}'")
                            event["plate_text"] = new_text
                            
                    return True, event
                else:
                    logger.debug(f"LLM Rejected event: {reason}")
                    return False, None
            except json.JSONDecodeError as exc:
                logger.warning(f"Verifier JSONDecodeError on text '{raw_text}': {exc}. Keeping event.")
                return True, event

        except Exception as e:
            logger.warning(f"Verification error (keeping event): {e}")
            return True, event # Fail open (keep result) on error

    async def verify_results(self, query, results, max_concurrency=20):
        """
        Main entry point. Filters the list of results using LLM asynchronously.
        
        Args:
            query (str): The original user query (e.g. "red car", "gun").
            results (list): List of event dictionaries containing 'thumbnail' and 'bbox'.
            max_concurrency (int): Semaphore limit for concurrent LLM calls.
            
        Returns:
            list: Filtered list of verified results.
        """
        if not self.enabled or not results:
            return results

        logger.info(f"LLM Verifying {len(results)} results for query: '{query}'...")
        start_time = time.time()
        
        verified_results = []
        sem = asyncio.Semaphore(max_concurrency)
        
        async def bounded_verify(event):
            async with sem:
                return await self._verify_single_event(event, query)

        tasks = [bounded_verify(event) for event in results]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for outcome in outcomes:
            if isinstance(outcome, Exception):
                logger.error(f"Task failed during async verification: {outcome}")
                continue
            kept, updated_event = outcome
            if kept and updated_event:
                verified_results.append(updated_event)

        # Sort back to time order if needed (optional)
        verified_results.sort(key=lambda x: x.get("time_sec", 0))

        elapsed = time.time() - start_time
        dropped = len(results) - len(verified_results)
        logger.info(f"Verification done in {elapsed:.2f}s. Kept {len(verified_results)}, Dropped {dropped}.")
        
        return verified_results
