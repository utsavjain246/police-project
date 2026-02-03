
# detectors.py

import time
import json
import logging
import cv2
import requests
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLM_Lite:
    """
    VLM using Custom API Endpoint.
    Sends frames as 'image' multipart/form-data to the user's endpoint.
    """
    def __init__(self):
        logger.info(f"Initializing VLM_Lite (Endpoint: {config.VLM_ENDPOINT})...")
        self.endpoint = config.VLM_ENDPOINT
        
        # Use a Session for Keep-Alive (reduces SSL handshakes)
        self.session = requests.Session()
        
        # Add Retry Logic
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def describe_scene(self, frame_array):
        """
        Encodes frame as JPEG and POSTs to the custom VLM endpoint.
        """
        start = time.time()
        try:
            # 1. Resize Frame (Downscale for speed)
            # Use dimensions from config
            frame_resized = cv2.resize(frame_array, (config.VLM_RESIZE_WIDTH, config.VLM_RESIZE_HEIGHT))
            
            # 2. Encode Frame to JPEG (Default Quality)
            ret, buffer = cv2.imencode('.jpg', frame_resized)
            if not ret:
                logger.error("Failed to encode frame")
                return {"error": "frame_encoding_failed"}
            
            # 2. Prepare Payload
            # Based on user request: Prompt is handled on backend. Sending image only.
            
            files = {
                'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')
            }
            # No data/prompt field sent
            
            # 3. Request (using Session)
            response = self.session.post(self.endpoint, files=files, timeout=30)
            
            # 4. Parse Response
            if response.status_code == 200:
                try:
                    raw_result = response.json()
                    logger.info(f"VLM Response ({time.time()-start:.2f}s): {raw_result}")
                    
                    # Handle OpenAI-compatible format (choices -> message -> content)
                    if "choices" in raw_result and len(raw_result["choices"]) > 0:
                        content = raw_result["choices"][0]["message"]["content"]
                        
                        # Strip markdown if present
                        if "```json" in content:
                            content = content.replace("```json", "").replace("```", "")
                        elif "```" in content:
                            content = content.replace("```", "")
                            
                        # Parse the inner JSON string
                        try:
                            parsed_content = json.loads(content.strip())
                            return parsed_content
                        except json.JSONDecodeError:
                            return {"description": content, "alert": False, "raw": True}
                    
                    # Direct JSON format fallback
                    return raw_result
                    
                except json.JSONDecodeError:
                    return {"description": response.text, "alert": False, "raw": True}
            else:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return {"error": f"api_error_{response.status_code}", "desc": response.text}
                
        except Exception as e:
            logger.error(f"VLM Connect Error: {e}")
            return {
                "description": "Connection Error",
                "alert": False,
                "error": str(e)
            }

class GeminiVLM:
    """
    VLM using Google Gemini 1.5 Flash.
    """
    def __init__(self):
        logger.info(f"Initializing GeminiVLM ({config.GEMINI_MODEL})...")
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        if not config.GEMINI_API_KEY or "YOUR_API_KEY" in config.GEMINI_API_KEY:
            logger.error("Gemini API Key not set! Please check config.py")
            raise ValueError("GEMINI_API_KEY_MISSING")
            
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        
        self.generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=300,
            temperature=0.4,
            response_mime_type="application/json"
        )
        
        # Block NOTHING - Surveillance needs to see violence potential
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.prompt = """
        You are a Police Surveillance AI. 
        TASK: Analyze the image for IMMEDIATE security threats.
        OUTPUT: Strict JSON only. No markdown formatting. No explanation text outside JSON.

        INSTRUCTIONS:
        - Analyze the image for these specific threats.
        - Ignore normal behavior (walking, standing, talking).
        - Be decisive. If unsafe, flag it.

        CATEGORIES:
        1. VIOLENCE: Fighting, striking, physical assault.
        2. WEAPONS: Visible guns, knives, bats.
        3. ACCIDENT: Vehicle crashes, fire, smoke, fallen person.
        4. TRAFFIC: No helmet, triple riding, PARKING VIOLATION (vehicle in no-parking zone/blocking road).
        5. SUSPICIOUS: crowd panic.

        JSON SCHEMA:
        {
          "threat_detected": boolean, 
          "category": "String (One of the categories above)",
          "severity": "LOW" | "MEDIUM" | "HIGH",
          "description": "Max 5 words identifying the specific threat",
          "objects": ["List of relevant objects"] 
        }

        NOTE: If the scene is normal, output {"threat_detected": false, "category": "SAFE", ...}.
        """


    def describe_scene(self, frame_array):
        """
        Sends frame to Gemini.
        """
        start = time.time()
        try:
            # 1. Resize Frame
            frame_resized = cv2.resize(frame_array, (config.VLM_RESIZE_WIDTH, config.VLM_RESIZE_HEIGHT))
            
            # 2. Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', frame_resized)
            if not ret:
                return {"error": "frame_encoding_failed"}
            
            from PIL import Image
            import io
            pil_image = Image.open(io.BytesIO(buffer))
            
            # 3. Generate
            try:
                response = self.model.generate_content(
                    [self.prompt, pil_image],
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
            except Exception as e:
                if "429" in str(e):
                    logger.warning("Gemini 429 Rate Limit Hit")
                    time.sleep(5) # Cooldown
                    return {"error": "rate_limit", "alert": False}
                raise e
            
            # 4. Parse Response Safely
            try:
                # Check validation (Finish Reason)
                if response.candidates and response.candidates[0].finish_reason != 1: # 1 = STOP
                     # 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION
                    reason = response.candidates[0].finish_reason
                    logger.warning(f"Gemini Finish Reason: {reason}")
                    if reason == 3: # Safety
                        return {"description": "Blocked by Safety Filters", "alert": True, "type": "Blocked"}
                
                text_content = response.text
                
                # Strip overrides
                if "```json" in text_content:
                    text_content = text_content.replace("```json", "").replace("```", "")
                elif "```" in text_content:
                    text_content = text_content.replace("```", "")
                
                result = json.loads(text_content)
                logger.info(f"Gemini Response ({time.time()-start:.2f}s): {result}")
                return result
                
            except ValueError:
                # response.text access failed
                logger.error(f"Gemini Response Invalid: {response.candidates}")
                return {"description": "Invalid Response", "alert": False}
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Gemini JSON: {text_content}")
                return {"description": text_content, "alert": False, "raw": True}
                
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return {"error": str(e), "alert": False}

    def detect(self, frame, target_object):
        return {}
