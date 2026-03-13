import time
import multiprocessing
import queue
import os
import cv2
from multiprocessing import Process, Event
import config
from ring_buffer import RingBuffer
from queues import DropOldestQueue, DropNewestQueue
from detectors import VLM_Lite

def vlm_worker_logic(input_queue, result_queue, stop_event):
    """
    The worker process running the VLM-First logic.
    """
    print("VLM Worker: Starting...")
    
    # Connect to Shared Memory Ring Buffer
    ring_buffer = RingBuffer(create=False)
    
    # Initialize Models
    # Initialize Models
    vlm = VLM_Lite()
    #from detectors import GeminiVLM
    #vlm = GeminiVLM()
    #yolo = YOLO_Tiny()

    
    # Ensure Alerts Dir exists
    os.makedirs(config.ALERTS_DIR, exist_ok=True)
    
    while not stop_event.is_set():
        try:
            # 1. Get Frame Metadata (Blocking with timeout to check stop_event)
            frame_meta = input_queue.get() 
            
            # 2. Read Frame from Shared Memory (Zero-Copy-ish)
            frame, success = ring_buffer.read_safe(frame_meta)
            
            if not success:
                # Frame overwritten, skip
                continue
            
            # 3. Step 1: VLM Description (CRITICAL: First step)
            start_time = time.time()
            scene_analysis = vlm.describe_scene(frame)
            
            # 4. Step 2: Decision Logic
            # Check for 'alert' (our prompt) OR 'threat_detected' (server default prompt)
            is_alert = scene_analysis.get("alert", False) or scene_analysis.get("threat_detected", False)
            
            final_output = {
                "frame_id": frame_meta.frame_id,
                "timestamp": frame_meta.timestamp,
                "analysis_latency": time.time() - start_time,
                "vlm_desc": scene_analysis["description"],
                "alert": is_alert,
                "actions_taken": ["vlm_describe"]
            }
            
            # HANDLE ALERT SAVING
            if is_alert:
                timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(frame_meta.timestamp))
                filename = f"alert_frame_{frame_meta.frame_id}_{timestamp_str}.jpg"
                save_path = os.path.join(config.ALERTS_DIR, filename)
                
                # Save the frame
                cv2.imwrite(save_path, frame)
                final_output["snapshot_path"] = save_path
                final_output["actions_taken"].append("snapshot_saved")
            
            # If VLM says "vehicle", maybe we run YOLO to get the plate or box?
            if scene_analysis.get("type") == "vehicle":
                det_result = yolo.detect(frame, "car")
                final_output["yolo_result"] = det_result
                final_output["actions_taken"].append("yolo_verify")
            
            # 5. Push to Result Queue
            result_queue.put(final_output)
            
        except BrokenPipeError:
            break
        except Exception as e:
            # print(f"Worker Error: {e}")
            pass
            
    print("VLM Worker: Stopping...")
    ring_buffer.close()

class RealtimeOrchestrator:
    def __init__(self):
        self.stop_event = Event()
        
        # Infrastructure
        self.ring_buffer = RingBuffer(create=True)
        self.input_queue = DropOldestQueue(maxsize=config.INPUT_QUEUE_SIZE)
        self.result_queue = multiprocessing.Queue(maxsize=config.RESULT_QUEUE_SIZE)
        
        # Workers
        self.vlm_processes = []
        for i in range(config.VLM_WORKER_COUNT):
            p = Process(
                target=vlm_worker_logic,
                args=(self.input_queue, self.result_queue, self.stop_event),
                name=f"VLM_Worker_{i}"
            )
            self.vlm_processes.append(p)
        
    def start(self):
        for p in self.vlm_processes:
            p.start()
        
    def stop(self):
        self.stop_event.set()
        for p in self.vlm_processes:
            p.join()
        self.ring_buffer.close()
        self.ring_buffer.unlink()
        print("Orchestrator stopped.")

    def ingest_frame(self, frame_data, frame_id):
        """
        Public API: Buffers frames and bursts them to the queue every 3 seconds.
        """
        # 1. Buffer the frame
        # We need to store data + id to write later, or write now and store meta?
        # Writing to RingBuffer is cheap. Let's write NOW, but only Push Meta LATER if selected.
        
        # Write to Ring Buffer (Always write to keep 'latest' available if needed?)
        # Actually, if we write 90 frames, we wrap around a 100-slot buffer instantly?
        # We increased RingBuffer to 200 slots. Safe for 90 frames.
        
        # Strategy: Write every frame to Ring Buffer? 
        # No, that churns the buffer unnecessarily if we only need 6.
        # Let's Buffer in valid memory (RAM list) then Write only selected ones.
        
        if not hasattr(self, 'frame_buffer'):
            self.frame_buffer = []

        self.frame_buffer.append((frame_data, frame_id))

        # 2. Check Batch Completion
        if len(self.frame_buffer) >= config.BATCH_SIZE_FRAMES:
            # Batch Full (3s / 90 frames). Process.
            
            # Select every 15th frame (indices 14, 29, 44...)
            # "Every 15th frame" -> 15th, 30th... (1-indexed count) -> indices 14, 29...
            # Or 0, 15, 30... (0-indexed). Let's do 0, 15, 30...
            selected_indices = range(0, len(self.frame_buffer), config.SAMPLE_EVERY_N_FRAMES)
            
            print(f"DEBUG: Processing Batch {frame_id}. Selected {len(selected_indices)} frames.")
            
            for idx in selected_indices:
                f_data, f_id = self.frame_buffer[idx]
                
                # Write to Ring Buffer
                meta = self.ring_buffer.write(f_data, f_id)
                
                # Push to Queue
                self.input_queue.put(meta)
                
            # 3. Clear Buffer
            self.frame_buffer = []
        
    def get_results(self):
        """
        Get results from the pipeline (Non-blocking)
        """
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results
