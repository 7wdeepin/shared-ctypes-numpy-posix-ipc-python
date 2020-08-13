import mmap
import logging
import numpy as np
from ctypes import sizeof, memmove, addressof, create_string_buffer

import posix_ipc as ipc
from shm.structures import MD

# Create a buffer initialized to NUL bytes for metadata
md_buf = create_string_buffer(sizeof(MD))


class SharedMemoryFrameWriter:

    def __init__(self, name):
        logging.info("Writer launched")

        # Name of shared memory for frame
        self.SHM_NAME_FR = name
        # Name of shared memory for metadata
        self.SHM_NAME_MD = name + "_md"
        # Name of semaphore for writing operation
        self.SEM_NAME_RD = name + "_rd"
        # Name of semaphore for reading operation
        self.SEM_NAME_WR = name + "_wr"

        # Map metadata into shared memory
        self.shm_md = ipc.SharedMemory(name=self.SHM_NAME_MD, flags=ipc.O_CREAT, size=sizeof(MD))
        self.map_md = mmap.mmap(self.shm_md.fd, self.shm_md.size)
        self.shm_md.close_fd()

        # Map frame into shared memory later
        self.shm_fr = None
        self.map_fr = None

        # Create two binary semaphores for read/write synchronization
        try:
            # Semaphore for reading
            self.sem_r = ipc.Semaphore(name=self.SEM_NAME_RD, flags=ipc.O_CREX, mode=0o666, initial_value=0)

            # Semaphore for writing
            self.sem_w = ipc.Semaphore(name=self.SEM_NAME_WR, flags=ipc.O_CREX, mode=0o666, initial_value=1)

        except ipc.ExistentialError:
            # Semaphore for reading
            sem_r = ipc.Semaphore(name=self.SEM_NAME_RD, flags=ipc.O_CREAT, mode=0o666, initial_value=0)
            sem_r.unlink()
            self.sem_r = ipc.Semaphore(name=self.SEM_NAME_RD, flags=ipc.O_CREX, mode=0o666, initial_value=0)

            # Semaphore for writing
            sem_w = ipc.Semaphore(name=self.SEM_NAME_WR, flags=ipc.O_CREAT, mode=0o666, initial_value=1)
            sem_w.unlink()
            self.sem_w = ipc.Semaphore(name=self.SEM_NAME_WR, flags=ipc.O_CREX, mode=0o666, initial_value=1)

    def add(self, frame: np.ndarray):
        self.sem_w.acquire()

        # Map frame into shared memory 
        byte_size = frame.nbytes
        if not self.shm_fr:
            self.shm_fr = ipc.SharedMemory(name=self.SHM_NAME_FR, flags=ipc.O_CREAT, size=byte_size)
            self.map_fr = mmap.mmap(self.shm_fr.fd, byte_size)
            self.shm_fr.close_fd()

        # Write metadata to shared memory
        md = MD(frame.shape[0], frame.shape[1], frame.shape[2], byte_size)
        memmove(md_buf, addressof(md), sizeof(md))
        self.map_md[:] = bytes(md_buf)

        # Write frame to shared memory
        self.map_fr[:] = frame.tobytes()

        self.sem_r.release()

    def release(self):
        self.map_md.close()
        ipc.unlink_shared_memory(self.SHM_NAME_MD)

        self.map_fr.close()
        ipc.unlink_shared_memory(self.SHM_NAME_FR)

        self.sem_w.close()
        self.sem_r.close()

        logging.info("Writer terminated")
