"""Part 1: Experiment A - The Stubborn Serial Tail.

Compute work is split cleanly across worker processes, but the master first
sleeps 0.40 s (an intentional sequential bottleneck). Karp-Flatt e(p) should
stay flat across P.

The i9-13900HX is hybrid (8 P-cores + 16 E-cores). To keep Windows from moving
workers onto the slower E-cores mid-run, the process is pinned to one hardware
thread per P-core (logical CPUs 0,2,...,14); worker processes inherit it.
Each P is measured REPS times and the median is reported.
"""
import json
import math
import statistics
import time
from multiprocessing import Pool

import psutil

P_CORES = list(range(0, 16, 2))
REPS = 3


def heavy_math(n):
    return sum(math.sin(i) * math.cos(i) for i in range(1, n))


def run_experiment(workers, total_chunks=16, chunk_size=2_000_000):
    t0 = time.perf_counter()
    time.sleep(0.40)  # Intentional sequential bottleneck (~12% serial fraction)
    with Pool(processes=workers) as pool:
        pool.map(heavy_math, [chunk_size] * total_chunks)
    return time.perf_counter() - t0


if __name__ == "__main__":
    psutil.Process().cpu_affinity(P_CORES)
    cores = [1, 2, 4]
    runtimes = {p: statistics.median(run_experiment(p) for _ in range(REPS)) for p in cores}
    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s")
    for p in [2, 4]:
        tp = runtimes[p]
        sp = t1 / tp
        e = ((p * tp / t1) - 1.0) / (p - 1.0)
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | Karp-Flatt e={e:.3f}")
    with open("exp_a_results.json", "w") as f:
        json.dump(runtimes, f, indent=2)
