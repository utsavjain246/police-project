
# queues.py

import queue
from multiprocessing import Queue as ProcessQueue

class DropOldestQueue:
    """
    Queue that drops the oldest item when full to make room for new items.
    Used for the Input Queue (Frames) to ensure we always process the latest video.
    """
    def __init__(self, maxsize):
        self.queue = ProcessQueue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            # Queue is full. Discard the oldest item to make space.
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass # Rare race condition, ignore
            
            # Try putting again. If it fails again, we just drop this new frame (backpressure fallback)
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                pass # Dropping latest as fallback

    def get(self):
        return self.queue.get()

    def empty(self):
        return self.queue.empty()


class DropNewestQueue:
    """
    Queue that drops the NEWEST incoming item if full.
    Used for Detectors/Analysis if they are too slow (shed load).
    """
    def __init__(self, maxsize):
        self.queue = ProcessQueue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            # Queue is full. Drop this new item.
            # Do nothing.
            pass

    def get(self):
        return self.queue.get()
        
    def empty(self):
        return self.queue.empty()
