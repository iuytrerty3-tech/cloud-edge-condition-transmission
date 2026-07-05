from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, TypeVar

import torch


T = TypeVar("T")


def simulate_network_time_seconds(num_bytes: int, bandwidth_mbps: float, rtt_ms: float = 0.0) -> float:
    bytes_per_second = bandwidth_mbps * 1024 * 1024 / 8.0
    return (num_bytes / bytes_per_second) + (rtt_ms / 1000.0)


def format_bytes(num_bytes: int) -> dict[str, float]:
    kb = num_bytes / 1024.0
    mb = kb / 1024.0
    return {"bytes": float(num_bytes), "kb": kb, "mb": mb}


def measure_cuda_seconds(fn: Callable[[], T], device: str = "cuda") -> tuple[T, float]:
    if not torch.cuda.is_available() or "cuda" not in device:
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        return result, elapsed

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    torch.cuda.synchronize()
    return result, start_event.elapsed_time(end_event) / 1000.0


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
