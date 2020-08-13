import mmap
import logging
import numpy as np
from time import sleep
from ctypes import sizeof, memmove, addressof, create_string_buffer

import posix_ipc as ipc
from shm.structures import MD

# Create a buffer initialized to NUL bytes for metadata
md_buf = create_string_buffer(sizeof(MD))


class SharedMemoryFrameReader:
    """Frame reader of shared memory
    Note: all shared memory segments and semaphores are created by frame writer.
    """

    def __init__(self, name):
        logging.info("Reader launched")

        # Name of shared memory for frame
        self.SHM_NAME_FR = name
        # Name of shared memory for metadata
        self.SHM_NAME_MD = name + "_md"
        # Name of semaphore for writing operation
        self.SEM_NAME_RD = name + "_rd"
        # Name of semaphore for reading operation
        self.SEM_NAME_WR = name + "_wr"

        # With flags set to the default of 0, 
        # the module attempts to open an existing segment 
        # and raises an error if that segment doesn't exist.
        # Wait for shared memory of metadata available 
        self.map_md = None
        while not self.map_md:
            try:
                logging.warning("Waiting for shared memory of metadata available.")
                shm_md = ipc.SharedMemory(name=self.SHM_NAME_MD, flags=0)
                self.map_md = mmap.mmap(shm_md.fd, sizeof(MD))
                shm_md.close_fd()
            except ipc.ExistentialError:
                sleep(1)

        # Map frame into shared memory later
        self.map_fr = None

        # With flags set to the default of 0, 
        # the module attempts to open an existing semaphore 
        # and raises an error if that semaphore doesn't exist.
        # Semaphore for reading
        self.sem_r = None
        while not self.sem_r:
            try:
                logging.warning("Waiting for semaphore of reading available.")
                self.sem_r = ipc.Semaphore(name=self.SEM_NAME_RD, flags=0)
            except ipc.ExistentialError:
                sleep(1)

        # Semaphore for writing
        self.sem_w = None
        while not self.sem_w:
            try:
                logging.warning("Waiting for semaphore of writing available.")
                self.sem_w = ipc.Semaphore(name=self.SEM_NAME_WR, flags=0)
            except ipc.ExistentialError:
                sleep(1)

    def get(self):
        self.sem_r.acquire()

        # Deserialize metadata
        md = MD()
        md_buf[:] = self.map_md
        memmove(addressof(md), md_buf, sizeof(md))

        # Map frame into shared memory 
        if not self.map_fr:
            try:
                shm_fr = ipc.SharedMemory(name=self.SHM_NAME_FR, flags=0)
                self.map_fr = mmap.mmap(shm_fr.fd, md.size)
                shm_fr.close_fd()
            except ipc.ExistentialError:
                logging.error("The shared memory of frame does not exist.")
                return None

        # Deserialize frame
        frame = np.ndarray((md.height, md.width, md.channels), dtype='uint8', 
                           buffer=self.map_fr)

        self.sem_w.release()
        return frame

    def release(self):
        self.map_md.close()
        self.map_fr.close()

        self.sem_w.close()
        self.sem_r.close()

        logging.info("Reader terminated")
