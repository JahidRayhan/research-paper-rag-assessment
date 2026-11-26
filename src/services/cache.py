"""Tiny in-memory TTL cache for repeated queries (cleared on upload/delete)."""
import threading
import time


class TTLCache:
    def __init__(self, ttl: int, maxsize: int = 256):
        self.ttl, self.maxsize = ttl, maxsize
        self._d: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._d.get(key)
            if not item:
                return None
            if time.time() - item[0] > self.ttl:
                del self._d[key]
                return None
            return item[1]

    def set(self, key, value) -> None:
        with self._lock:
            if len(self._d) >= self.maxsize:
                self._d.pop(next(iter(self._d)))
            self._d[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
