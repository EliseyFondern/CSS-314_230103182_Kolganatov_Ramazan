"""Part 1: Experiment A - The Granularity Trap.

The same Collatz workload is mapped over P workers twice: chunksize=1 (one
IPC task envelope per number, micro-tasks) and chunksize=10000 (amortized
chunks). Fine-grained dispatch should starve on the master process.

The i9-13900HX is hybrid (8 P-cores + 16 E-cores). The process is pinned to
the P-core hardware threads (logical CPUs 0-15) so Windows does not move
workers onto the slower E-cores; worker processes inherit the mask.
Each P is measured REPS times and the median is reported.
"""
import json
import statistics
import time
from multiprocessing import Pool

import psutil

P_CORES = list(range(16))
REPS = 3
CORES = [1, 2, 4, 8]


def collatz_steps(n):
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def run_granularity(workers, total_items=1_000_000, chunk_size=1):
    numbers = range(1, total_items + 1)
    t0 = time.perf_counter()
    with Pool(processes=workers) as pool:
        # chunk_size=1 creates 1,000,000 individual IPC task envelopes (Micro-tasks)
        results = pool.map(collatz_steps, numbers, chunksize=chunk_size)
    return time.perf_counter() - t0, sum(results)


def measure(chunk_size):
    runtimes, checksums = {}, set()
    for p in CORES:
        runs = [run_granularity(p, chunk_size=chunk_size) for _ in range(REPS)]
        runtimes[p] = statistics.median(t for t, _ in runs)
        checksums.update(s for _, s in runs)
    assert len(checksums) == 1, checksums
    return runtimes, checksums.pop()


def report(title, runtimes):
    t1 = runtimes[1]
    print(title)
    print(f"  P= 1 | Time={t1:6.3f}s")
    for p in CORES[1:]:
        tp = runtimes[p]
        sp = t1 / tp
        e = ((p * tp / t1) - 1.0) / (p - 1.0)
        print(f"  P={p:2d} | Time={tp:6.3f}s | Speedup={sp:5.2f}x | Efficiency={sp / p:4.0%} | Karp-Flatt e={e:6.3f}")


if __name__ == "__main__":
    psutil.Process().cpu_affinity(P_CORES)
    fine, sum_fine = measure(chunk_size=1)
    coarse, sum_coarse = measure(chunk_size=10_000)
    assert sum_fine == sum_coarse
    print(f"Collatz checksum (both granularities): {sum_fine}\n")
    report("chunksize=1 (micro-tasks):", fine)
    report("chunksize=10000 (amortized chunks):", coarse)
    with open("exp_a_results.json", "w") as f:
        json.dump({"checksum": sum_fine, "chunksize_1": fine, "chunksize_10000": coarse}, f, indent=2)
