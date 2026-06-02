import time
import os


class Timer:
    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


class ResourceMonitor:
    def __init__(self, framework="sklearn"):
        self.framework = framework
        self.peak_cpu_mb = 0
        self.peak_gpu_mb = 0
        self.timer = Timer()
        self._process = None

    def _get_process(self):
        if self._process is None:
            import psutil
            self._process = psutil.Process(os.getpid())
        return self._process

    def start(self):
        self.timer.__enter__()
        return self

    def stop(self):
        self.timer.__exit__(None, None, None)
        proc = self._get_process()
        try:
            mem_info = proc.memory_info()
            self.peak_cpu_mb = mem_info.rss / 1024 / 1024
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        except (ImportError, RuntimeError):
            pass

    def summary(self):
        return {
            "framework": self.framework,
            "train_time_s": round(self.timer.elapsed, 2) if self.timer.elapsed else 0,
            "peak_cpu_mb": round(self.peak_cpu_mb, 1),
            "peak_gpu_mb": round(self.peak_gpu_mb, 1),
        }
