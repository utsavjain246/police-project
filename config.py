
# config.py

import os

# Frame Settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CHANNELS = 3
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, CHANNELS)
FRAME_DTYPE = 'uint8'
FRAME_SIZE_BYTES = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS

# Ring Buffer Settings
RING_BUFFER_SLOTS = 200 
SHARED_MEMORY_NAME = "video_orchestrator_shm"

# Queue Settings
INPUT_QUEUE_SIZE = 20
RESULT_QUEUE_SIZE = 100

# VLM Settings
VLM_ENDPOINT = "https://unspayed-ruefully-neville.ngrok-free.dev/vlm"
# Optimization: Increase resolution for accuracy (512x384)
VLM_RESIZE_WIDTH = 640
VLM_RESIZE_HEIGHT = 480

# Gemini Settings
GEMINI_API_KEY = "AIzaSyBBGdd_E5z7yAIQ9OhGQIJIFIMFUI3g1Gk"
GEMINI_MODEL = "gemini-flash-lite-latest"


# Batch Sampling Strategy
FPS = 30
BATCH_DURATION_SEC = 3.0
BATCH_SIZE_FRAMES = int(FPS * BATCH_DURATION_SEC) 
SAMPLE_EVERY_N_FRAMES = 15 

# Parallelism
VLM_WORKER_COUNT = 2

# Alerting
ALERTS_DIR = "alerts"
