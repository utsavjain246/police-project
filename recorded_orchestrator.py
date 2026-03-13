
import os
import json
import logging
from detectors_forensic import LicensePlateSkill, CrowdCountSkill, GeneralEventSkill
from detectors import GeminiVLM

logger = logging.getLogger(__name__)

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
        if not self.vlm:
            return self.heuristic_fallback(query)

        prompt = f"""
        You are the Brain of a Forensic Video Analysis System.
        USER QUERY: "{query}"
        
        Available Tools:
        1. LPR: Use ONLY for "license plate", "vehicle number", "read plate" queries.
        2. CROWD: Use ONLY for "count people", "crowd density", "how many people".
        3. GENERAL: Use for EVERYTHING else (fighting, accidents, red car, searching objects).
        
        TASK: Return valid JSON only.
        {{
            "tool": "LPR" | "CROWD" | "GENERAL",
            "target": "String (The specific thing to look for, e.g. 'fighting' or 'DL8C')",
            "params": {{ "min_confidence": 0.5 }}
        }}
        """
        
        try:
            # We use the VLM to generate content from text prompt only (no image)
            # Gemini 1.5 Flash supports text-only prompts
            response = self.vlm.model.generate_content(prompt)
            
            text = response.text
            if "```json" in text:
                text = text.replace("```json", "").replace("```", "")
            elif "```" in text:
                 text = text.replace("```", "")
                 
            config = json.loads(text.strip())
            return config
            
        except Exception as e:
            logger.error(f"Intent Analysis Failed: {e}")
            return self.heuristic_fallback(query)

    def heuristic_fallback(self, query):
        query = query.lower()
        if any(x in query for x in ["plate", "vehicle", "license", "car number"]):
            return {"tool": "LPR", "target": query}
        elif any(x in query for x in ["count", "crowd", "people"]):
            if "fight" in query or "accident" in query:
                 return {"tool": "GENERAL", "target": query}
            return {"tool": "CROWD", "target": query}
        else:
            return {"tool": "GENERAL", "target": query}

class RecordedOrchestrator:
    def __init__(self):
        self.skills = {
            "LPR": LicensePlateSkill(),
            "CROWD": CrowdCountSkill(),
            "GENERAL": GeneralEventSkill()
        }
        self.analyzer = QueryAnalyzer()
    
    def process(self, video_path, query):
        print(f"Analyzing Query: '{query}'...")
        
        # 1. Analyze Intent
        task_config = self.analyzer.analyze(query)
        tool_name = task_config.get("tool", "GENERAL")
        target_param = task_config.get("target", query)
        
        print(f"Intent Classified: Tool=[{tool_name}] Target=['{target_param}']")
        
        # 2. Select Skill
        skill = self.skills.get(tool_name)
        if not skill:
            return {"error": f"Tool {tool_name} not implemented"}
            
        # 3. Execute
        print(f"Starting {tool_name} analysis on {os.path.basename(video_path)}...")
        try:
            results = skill.process_video(video_path, target_param)
        except Exception as e:
            print(f"Error during execution: {e}")
            return {"error": str(e)}
        
        # 4. Generate Report
        report = {
            "query": query,
            "video": video_path,
            "tool_used": tool_name,
            "intent_config": task_config,
            "events_found": len(results),
            "results": results
        }
        
        # Save Report
        with open("forensic_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        return report

if __name__ == "__main__":
    # Test Stub
    orc = RecordedOrchestrator()
    # orc.process("video.mp4", "Find plate DL8C")
