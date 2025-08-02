import time

class RateLimiter:
    def __init__(self, calls_per_minute):
        self.delay = 60 / calls_per_minute
        self.last_call = 0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call = time.time()
