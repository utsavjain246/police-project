import os
import json
import logging
import numpy as np
import config
from detectors_forensic import (
    LicensePlateSkill,
    CrowdCountSkill,
    GeneralEventSkill,
    WeaponDetectionSkill,
    VehicleColorSkill,
)
from detectors import GeminiVLM
from person_video_search import PersonVideoSearch

config.configure_logging()
logger = logging.getLogger(__name__)

def _sanitize_numpy(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

class QueryAnalyzer:
    """
    Classifies user intent using VLM/LLM to select the right tool and parameters.
    """
    def __init__(self):
        try:
            self.vlm = GeminiVLM()
        except Exception as e:
            logger.warning(f"Failed to init GeminiVLM for intent analysis, falling back to heuristic: {e}")
            self.vlm = None

    def analyze(self, query):
        logger.info("Analyzing intent for query: %s", query)
        if not self.vlm:
            msg = f"--- Routing via HEURISTIC FALLBACK (No VLM initialized) ---"
            print(msg)
            logger.info(msg)
            return self.heuristic_fallback(query)

        prompt = f"""
        You are the Brain of a Forensic Video Analysis System.
        USER QUERY: "{query}"
        
        Available Tools:
        1. LPR: Use ONLY for "license plate", "vehicle number", "read plate" queries.
        2. CROWD: Use ONLY for "count people", "crowd density", "how many people".
        3. WEAPON: Use ONLY for "guns", "knives", "weapons", "pistols", "bat".
        4. VEHICLE_COLOR: Use ONLY for color + vehicle queries like "red car", "blue truck".
        5. PERSON: Use ONLY for strict facial recognition or identity searches (e.g., "find this specific person", "face search", "identify him").
        6. GENERAL: Use for EVERYTHING else, including descriptive searches for people based on clothing/features (e.g., "find the person in the black shirt", "man with red hat"), as well as fighting, accidents, fire, etc.
        
        CRITICAL ROUTING RULE: If the query describes what a person is wearing or doing (e.g. "person in a black shirt", "man running"), it MUST go to GENERAL, not PERSON. PERSON is strictly for face-matching when a reference image is provided.
        
        TASK: Return valid JSON only.
        {{
            "tool": "LPR" | "CROWD" | "WEAPON" | "VEHICLE_COLOR" | "PERSON" | "GENERAL",
            "target": "String (The specific thing to look for, e.g. 'fighting', 'person in black shirt', or 'DL8C')",
            "params": {{ "min_confidence": 0.5, "max_results": 20 }}
        }}
        """
        
        try:
            # We use the VLM to generate content from text prompt only (no image)
            # Gemini 1.5 Flash supports text-only prompts
            response = self.vlm.model.generate_content(prompt, request_options={"timeout": 15.0})
            
            text = response.text
            if "```json" in text:
                text = text.replace("```json", "").replace("```", "")
            elif "```" in text:
                 text = text.replace("```", "")
                 
            config_json = json.loads(text.strip())
            msg = f"--- Routing via AI MODEL (VLM) for intent analysis ---"
            print(msg)
            logger.info(msg)
            logger.info("Intent analysis complete: %s", config_json)
            return config_json
            
        except Exception as e:
            logger.exception("Intent Analysis Failed: %s", e)
            msg = f"--- Routing via HEURISTIC FALLBACK (AI Model Error) ---"
            print(msg)
            logger.info(msg)
            return self.heuristic_fallback(query)

    def heuristic_fallback(self, query):
        logger.debug("Intent heuristic fallback for query: %s", query)
        query = query.lower()
        
        # Clothing/Descriptive person check overrides standard PERSON check
        clothing_keywords = ["shirt", "pants", "hat", "jacket", "wearing", "red", "black", "blue", "green", "white", "yellow"]
        is_descriptive_person = any(x in query for x in ["person", "man", "woman", "guy", "girl"]) and any(c in query for c in clothing_keywords)
        
        if is_descriptive_person:
            return {"tool": "GENERAL", "target": query}
            
        if any(color in query for color in ["red", "blue", "green", "yellow", "orange", "black", "white", "gray", "grey", "silver"]) and \
           any(v in query for v in ["car", "truck", "bus", "motorcycle", "bike", "vehicle", "van", "suv"]):
            return {"tool": "VEHICLE_COLOR", "target": query}
        if any(x in query for x in ["plate", "vehicle", "license", "car number"]):
            return {"tool": "LPR", "target": query}
        elif any(x in query for x in ["person", "face", "find him", "find her", "who is", "identify", "search person", "locate person"]):
            return {"tool": "PERSON", "target": query}
        elif any(x in query for x in ["count", "crowd", "people"]):
            if "fight" in query or "accident" in query:
                 return {"tool": "GENERAL", "target": query}
            return {"tool": "CROWD", "target": query}
        elif any(x in query for x in ["gun", "knife", "pistol", "rifle", "weapon"]):
            return {"tool": "WEAPON", "target": query}
        else:
            return {"tool": "GENERAL", "target": query}

class PersonSearchSkill:
    """Adapter that wraps PersonVideoSearch to match the orchestrator skill interface."""

    def __init__(self):
        self.engine = PersonVideoSearch()

    def process_video(self, video_path, target, **kwargs):
        reference_image = kwargs.get("reference_image")
        if not reference_image or not os.path.isfile(reference_image):
            logger.warning("PERSON tool invoked without a reference image")
            return None  # Caller handles this

        cancel_check = kwargs.get("cancel_check")
        
        # Extract params from kwargs (passed from the orchestrator intent parsing)
        params = kwargs.get("intent_config", {}).get("params", {})
        min_confidence = params.get("min_confidence", float(os.getenv("FACE_SEARCH_MIN_CONF", "0.2")))
        max_matches = params.get("max_results", int(os.getenv("FACE_SEARCH_MAX_MATCHES", "20")))

        report = self.engine.search(
            video_path=video_path,
            reference_image_path=reference_image,
            query_text=target or "",
            min_confidence=min_confidence,
            max_matches=max_matches,
            cooldown_sec=float(os.getenv("FACE_SEARCH_COOLDOWN_SEC", "0.5")),
        )

        # Normalize results to the event-list format the orchestrator expects
        raw_results = report.get("results", [])
        events = []
        for item in raw_results:
            events.append({
                "time_sec": item.get("time_sec", item.get("timestamp_sec", 0)),
                "timestamp": item.get("time_sec", item.get("timestamp_sec", 0)),
                "confidence": item.get("confidence", 0),
                "description": item.get("reason", "person match"),
                "method": item.get("method", "person_search"),
                "bbox": item.get("face_bbox"),
                "thumbnail": item.get("thumbnail"),
                "frame": item.get("frame_idx"),
            })
        return events


class RecordedOrchestrator:
    def __init__(self):
        logger.info("Initializing RecordedOrchestrator")
        self.skills = {
            "LPR": LicensePlateSkill(),
            "CROWD": CrowdCountSkill(),
            "WEAPON": WeaponDetectionSkill(),
            "VEHICLE_COLOR": VehicleColorSkill(),
            "PERSON": PersonSearchSkill(),
            "GENERAL": GeneralEventSkill(),
        }
        self.analyzer = QueryAnalyzer()
    
    def process(self, video_path, query, lpr_evidence_dir=None, reference_image=None, db=None, video_id=None, **_kwargs):
        logger.info("Analyzing query: %s", query)
        cancel_check = _kwargs.get("cancel_check")
        
        # 1. Analyze Intent
        task_config = self.analyzer.analyze(query)
        tool_name = task_config.get("tool", "GENERAL")
        target_param = task_config.get("target", query)
        
        logger.info("Intent classified: tool=%s target=%s", tool_name, target_param)
        
        # 2. Select Skill
        skill = self.skills.get(tool_name)
        if not skill:
            logger.error("Tool not implemented: %s", tool_name)
            return {"error": f"Tool {tool_name} not implemented"}

        # 3. Check domain aggregate cache
        aggregate = None
        from_cache = False
        if db and video_id:
            aggregate = db.get_domain_aggregate(video_id, tool_name)
            if aggregate:
                logger.info("Domain aggregate cache HIT for video=%s tool=%s — skipping scan", video_id, tool_name)
                from_cache = True

        # 4. Execute (only if no cache)
        if not from_cache:
            # PERSON tool requires a reference image
            if tool_name == "PERSON" and not reference_image:
                logger.info("PERSON tool selected but no reference image provided — returning text guidance")
                return {
                    "query": query,
                    "video": video_path,
                    "tool_used": "PERSON",
                    "intent_config": task_config,
                    "response_type": "text",
                    "events_found": 0,
                    "results": [],
                    "text_response": "To search for a specific person in the video, please upload a reference photo of the person you want to find. Use the face/person search feature and attach a clear photo of the person.",
                    "from_cache": False,
                }

            logger.info("Starting %s analysis on %s", tool_name, os.path.basename(video_path))
            try:
                skill_kwargs = {}
                # Pass intent_config into skill_kwargs so skills can read LLM-suggested parameters
                skill_kwargs["intent_config"] = task_config
                
                if cancel_check:
                    skill_kwargs["cancel_check"] = cancel_check
                if reference_image:
                    skill_kwargs["reference_image"] = reference_image
                if _kwargs.get("preprocessed_dir"):
                    skill_kwargs["preprocessed_dir"] = _kwargs.get("preprocessed_dir")
                
                if tool_name == "LPR" and lpr_evidence_dir:
                    skill_kwargs["plate_frame_dir"] = lpr_evidence_dir
                    skill_kwargs["save_plate_frames"] = True

                if tool_name in {"WEAPON", "CROWD"}:
                    results = skill.process_video(video_path, **skill_kwargs)
                else:
                    results = skill.process_video(video_path, target_param, **skill_kwargs)
            except Exception as e:
                logger.exception("Error during execution: %s", e)
                return {"error": str(e)}

            event_list = results if isinstance(results, list) else []

            # Build aggregate and cache it
            aggregate = {
                "events": event_list,
                "tool_used": tool_name,
            }
            if db and video_id:
                db.cache_domain_aggregate(video_id, tool_name, aggregate)
        else:
            event_list = aggregate.get("events", [])
            # Lazy-load thumbnails for cached events going to frontend
            for evt in event_list:
                db.load_thumbnail_b64(evt)

        # 5. Generate Report
        report = {
            "query": query,
            "video": video_path,
            "tool_used": tool_name,
            "intent_config": task_config,
            "events_found": len(event_list),
            "results": event_list,
            "from_cache": from_cache,
        }
        
        # Sanitize numpy types before serialization
        report = _sanitize_numpy(report)

        # Save Report
        with open("forensic_report.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Report saved: forensic_report.json (from_cache=%s)", from_cache)
            
        return report

if __name__ == "__main__":
    # Test Stub
    orc = RecordedOrchestrator()
    # orc.process("video.mp4", "Find plate DL8C")
