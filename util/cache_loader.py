import threading
import numpy as np
import os

class CacheLoader:
    def __init__(self, cache, data_dir):
        self.cache = cache
        self.data_dir = data_dir
        self.stop_signal = False
        self.thread = threading.Thread(target=self.load_data)
        self.thread.start()

    def load_data(self):
        while not self.stop_signal:
            npy_files_count = 0
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith('.npy'):
                        npy_files_count += 1
                        file_path = os.path.join(root, file)
                        if file_path not in self.cache.data:
                            data = np.load(file_path)
                            self.cache.put(file_path, data)
            if npy_files_count == len(self.cache.data):
                self.stop_signal = True
        self.thread.join()