import os
import time

class HeapQueue:
    def __init__(self, capacity):
        self.data = []
        self.capacity = capacity
    def enqueue(self, msg):
        if len(self.data) == self.capacity:
            self.data.pop(0)
        self.data.append(bytearray(msg))
    def flush(self):
        start = time.perf_counter()
        while self.data:
            self.data.pop(0)
        return time.perf_counter() - start

class RingBufferQueue:
    def __init__(self, capacity, item_size):
        self.buf = [bytearray(item_size) for _ in range(capacity)]
        self.sizes = [0]*capacity
        self.head = 0
        self.tail = 0
        self.count = 0
        self.capacity = capacity
        self.item_size = item_size
    def enqueue(self, msg):
        n = min(len(msg), self.item_size)
        self.buf[self.tail][:n] = msg[:n]
        self.sizes[self.tail] = n
        self.tail = (self.tail + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1
        else:
            self.head = (self.head + 1) % self.capacity
    def flush(self):
        start = time.perf_counter()
        self.head = (self.head + self.count) % self.capacity
        self.count = 0
        return time.perf_counter() - start

def benchmark():
    payload = os.urandom(512)
    heap_q = HeapQueue(1000)
    ring_q = RingBufferQueue(1000, 1024)
    for _ in range(1000):
        heap_q.enqueue(payload)
        ring_q.enqueue(payload)
    t1 = heap_q.flush()
    t2 = ring_q.flush()
    print(f"Heap-based flush: {t1:.6f}s")
    print(f"Ring buffer flush: {t2:.6f}s")

if __name__ == "__main__":
    benchmark()
