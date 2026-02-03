
# main.py

import sys
import time
import cv2
import os
import config
from orchestrator import RealtimeOrchestrator

def video_generator(source):
    """
    Generates frames from a real video file or webcam.
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        # Resize to config dimensions to match Shared Memory allocation
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        
        yield frame, frame_id
        
        frame_id += 1
        
        # Simulate realtime playback if reading from file
        # (Otherwise it reads as fast as possible)
        time.sleep(1/30) 

    cap.release()

def main():
    # 1. Get Video Source
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        # Check if it's a digit (webcam index)
        if video_source.isdigit():
            video_source = int(video_source)
    else:
        print("Usage: python main.py <video_path_or_cam_index>")
        print("Defaulting to Webcam (0)...")
        video_source = 0
        
    # 2. Check API Key for VLM
    # (Skipping check as we are using custom endpoint)

    print(f"Initializing Real-Time Video Orchestrator input={video_source}...")
    orchestrator = RealtimeOrchestrator()
    orchestrator.start()
    
    try:
        camera = video_generator(video_source)
        start_time = time.time()
        
        print(f"Pipeline Started. Reading from {video_source}. Press Ctrl+C to stop...")

        for frame, frame_id in camera:
            # 1. Ingest Frame
            orchestrator.ingest_frame(frame, frame_id)
            
            # 2. Check for Results (Non-blocking check)
            results = orchestrator.get_results()
            if results:
                for res in results:
                    latency = res.get('analysis_latency', 0)
                    desc = res.get('vlm_desc', 'No desc')
                    is_alert = res.get('alert', False)
                    
                    # Print formatted result
                    status = "ALERT!" if is_alert else "Monitor"
                    print(f"[{time.time()-start_time:.2f}s] FRAME {res['frame_id']} | Latency: {latency:.2f}s")
                    print(f"   >>> VLM: {desc}")
                    print(f"   >>> ACTION: {status}")
                    
                    if is_alert:
                        # Log to file
                        with open("alerts.log", "a", encoding="utf-8") as f:
                            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(res.get('timestamp', time.time())))
                            snap_path = res.get('snapshot_path', 'N/A')
                            log_entry = f"[{timestamp_str}] ALERT | Frame: {res['frame_id']} | Desc: {desc} | Img: {snap_path}\n"
                            f.write(log_entry)
                        print(f"       (Snapshot saved: {res.get('snapshot_path')})")
            
            # Optional: throttling loop to not overwhelm logging in terminal
            # time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        orchestrator.stop()

if __name__ == "__main__":
    main()
