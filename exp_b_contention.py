"""Part 2: Experiment B - The Contention Traffic Jam.

No serial sleep; every increment of one shared integer takes a mutex, so the
lock and the counter's cache line bounce between cores. Karp-Flatt e(p) should
surge with P.

Pinned to one hardware thread per P-core (logical CPUs 0,2,...,14), as in
Experiment A; median of REPS runs per P.
"""
import json
import statistics
import time
from multiprocessing import Process, Value, Lock

import psutil

P_CORES = list(range(0, 16, 2))
REPS = 3


def contended_worker(shared_counter, lock, iterations):
    for _ in range(iterations):
        with lock:  # Heavy lock contention & cache-line bouncing
            shared_counter.value += 1


def run_contended(p, total_increments=2_000_000):
    increments_per_worker = total_increments // p
    counter = Value('i', 0)
    lock = Lock()
    procs = [
        Process(target=contended_worker, args=(counter, lock, increments_per_worker))
        for _ in range(p)
    ]
    t0 = time.perf_counter()
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    t1 = time.perf_counter()
    assert counter.value == total_increments
    return t1 - t0


if __name__ == "__main__":
    psutil.Process().cpu_affinity(P_CORES)
    cores = [1, 2, 4]
    runtimes = {p: statistics.median(run_contended(p) for _ in range(REPS)) for p in cores}
    t1 = runtimes[1]
    print(f"P=1 | Runtime={t1:.3f}s")
    for p in [2, 4]:
        tp = runtimes[p]
        sp = t1 / tp
        e = ((p * tp / t1) - 1.0) / (p - 1.0)
        print(f"P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | Karp-Flatt e={e:.3f}")
    with open("exp_b_results.json", "w") as f:
        json.dump(runtimes, f, indent=2)
