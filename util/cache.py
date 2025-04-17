import threading

class Cache:
    def __init__(self):
        self.data = {}
        self.counter = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.counter[key] += 1
                if self.counter[key] >= 4 * 50:
                    value = self.data.pop(key)
                    self.counter.pop(key)
                else:
                    value = self.data[key]
            else:
                value = None
        return value

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.counter[key] = 0