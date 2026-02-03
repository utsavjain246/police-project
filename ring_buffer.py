
# ring_buffer.py

import sys
import time
import numpy as np
from multiprocessing import shared_memory, Lock
import config
from dataclasses import dataclass

@dataclass
class FrameMeta:
    """Metadata pointing to a frame in the ring buffer."""
    slot_id: int
    generation: int
    frame_id: int
    timestamp: float

class RingBuffer:
    """
    A shared memory ring buffer for zero-copy frame access.
    Uses generation counters to detect dropped/overwritten frames.
    """
    def __init__(self, create=False):
        self.slot_count = config.RING_BUFFER_SLOTS
        self.frame_size = config.FRAME_SIZE_BYTES
        self.shm_size = self.slot_count * self.frame_size
        self.shm_name = config.SHARED_MEMORY_NAME

        # Internal state tracking (process-local)
        self.write_index = 0
        self.lock = Lock() 
        # Note: In a true high-perf scenario, we might use atomic hardware primitives,
        # but a multiprocessing.Lock for the writer is fine for this scale.

        if create:
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=self.shm_size, name=self.shm_name)
            except FileExistsError:
                # Cleanup old memory if it exists
                temp_shm = shared_memory.SharedMemory(name=self.shm_name)
                temp_shm.close()
                temp_shm.unlink()
                self.shm = shared_memory.SharedMemory(create=True, size=self.shm_size, name=self.shm_name)
            
            # Initialize metadata table in a separate shared block or just manage in Python for now?
            # For simplicity in Python, we'll assume the READER receives the 'generation' 
            # via the FrameMeta passed through the Queue. 
            # BUT, to be robust against overwrites, the reader needs to verify the generation 
            # *after* reading from shared memory.
            # So we need a shared array for generations.
            
            self.gen_shm_name = self.shm_name + "_gen"
            self.gen_shm_size = self.slot_count * 8 # 64-bit integers
            try:
                self.gen_shm = shared_memory.SharedMemory(create=True, size=self.gen_shm_size, name=self.gen_shm_name)
            except FileExistsError:
                temp_shm = shared_memory.SharedMemory(name=self.gen_shm_name)
                temp_shm.close()
                temp_shm.unlink()
                self.gen_shm = shared_memory.SharedMemory(create=True, size=self.gen_shm_size, name=self.gen_shm_name)
                
            # Initialize generations to 0
            self.generations = np.ndarray((self.slot_count,), dtype='int64', buffer=self.gen_shm.buf)
            self.generations[:] = 0
            
        else:
            # Connect to existing
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            self.gen_shm = shared_memory.SharedMemory(name=self.shm_name + "_gen")
            self.generations = np.ndarray((self.slot_count,), dtype='int64', buffer=self.gen_shm.buf)

    def write(self, frame_data: np.ndarray, frame_id: int):
        """
        Writes a frame to the ring buffer.
        Returns FrameMeta to be passed to queues.
        """
        with self.lock:
            idx = self.write_index
            
            # 1. Update Generation (Pre-write) - odd number could indicate writing? 
            # Or just strictly increment.
            # We increment to indicate "this slot is changing".
            current_gen = self.generations[idx]
            new_gen = current_gen + 1
            self.generations[idx] = new_gen
            
            # 2. Write Data
            # Compute offset
            offset = idx * self.frame_size
            # Create numpy view into shared memory
            shm_view = np.ndarray(config.FRAME_SHAPE, dtype=config.FRAME_DTYPE, buffer=self.shm.buf, offset=offset)
            np.copyto(shm_view, frame_data)
            
            # 3. Update Generation (Post-write) - effectively commits the write?
            # Actually, the standard pattern is:
            # Gen initially even (0).
            # Writer: Gen += 1 (Odd, "writing"). Write. Gen += 1 (Even, "stable").
            # Reader: Check Gen (must be even). Read. Check Gen again (must be same).
            
            # Let's fix the logic above.
            self.generations[idx] += 1 # Done writing
            
            # Commit metadata
            meta = FrameMeta(
                slot_id=idx,
                generation=self.generations[idx], # The "stable" generation
                frame_id=frame_id,
                timestamp=time.time()
            )
            
            # Advance index
            self.write_index = (self.write_index + 1) % self.slot_count
            
            return meta

    def read_safe(self, meta: FrameMeta):
        """
        Reads a frame from the shared memory.
        Returns (frame, success).
        """
        idx = meta.slot_id
        expected_gen = meta.generation
        
        # 1. Check generation before read
        current_gen = self.generations[idx]
        if current_gen != expected_gen:
            # Slot has already been overwritten or is currently being written
            print(f"Frame dropped: Gen mismatch PRE (Exp: {expected_gen}, Act: {current_gen})")
            return None, False
            
        # 2. Read Data
        offset = idx * self.frame_size
        # We must copy to local memory to ensure we have a stable snapshot 
        # distinct from the changing shared memory
        shm_view = np.ndarray(config.FRAME_SHAPE, dtype=config.FRAME_DTYPE, buffer=self.shm.buf, offset=offset)
        frame_copy = shm_view.copy()
        
        # 3. Check generation after read
        post_gen = self.generations[idx]
        if post_gen != expected_gen:
            print(f"Frame dropped: Gen mismatch POST (Exp: {expected_gen}, Act: {post_gen})")
            return None, False
            
        return frame_copy, True

    def close(self):
        self.shm.close()
        self.gen_shm.close()

    def unlink(self):
        """Call this from the main process on shutdown"""
        self.shm.unlink()
        self.gen_shm.unlink()
