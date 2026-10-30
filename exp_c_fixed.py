"""Part 3: Experiment C - The Privatization Fix.

Golden Rule 8: privatize before synchronizing. Each worker counts into a
private local variable with zero locks; results are combined by one reduction
at the end. Karp-Flatt e(p) should collapse.

Pinned to one hardware thread per P-core (logical CPUs 0,2,...,14), as in
Experiments A and B; median of REPS runs per P.
"""
import json
import statistics
import time
from multiprocessing import Pool

import psutil

P_CORES = list(range(0, 16, 2))
REPS = 3


def private_worker(iterations):
    # Thread-local accumulation in private memory space (zero shared state)
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    return local_sum


def run_privatized(p, total_increments=20_000_000):
    increments_per_worker = total_increments // p
    tasks = [increments_per_worker] * p
    t0 = time.perf_counter()
    with Pool(processes=p) as pool:
        results = pool.map(private_worker, tasks)
    total = sum(results)  # Synchronize exactly ONCE at the end
    t1 = time.perf_counter()
    assert total == total_increments
    return t1 - t0


def run_privatized_warm(p, total_increments=20_000_000):
    # Same work, but the pool is started (and every worker warmed up) before
    # the clock starts: on Windows each worker is a freshly spawned interpreter,
    # and that one-off start-up cost is not part of the counting algorithm.
    increments_per_worker = total_increments // p
    tasks = [increments_per_worker] * p
    with Pool(processes=p) as pool:
        pool.map(private_worker, [1] * p, chunksize=1)
        t0 = time.perf_counter()
        results = pool.map(private_worker, tasks, chunksize=1)
        total = sum(results)  # Synchronize exactly ONCE at the end
        t1 = time.perf_counter()
    assert total == total_increments
    return t1 - t0


def report(title, runtimes):
    t1 = runtimes[1]
    print(title)
    print(f"  P=1 | Runtime={t1:.3f}s")
    for p in [2, 4]:
        tp = runtimes[p]
        sp = t1 / tp
        e = ((p * tp / t1) - 1.0) / (p - 1.0)
        print(f"  P={p} | Runtime={tp:.3f}s | Speedup={sp:.2f}x | Efficiency={sp / p:.0%} | Karp-Flatt e={e:.3f}")


if __name__ == "__main__":
    psutil.Process().cpu_affinity(P_CORES)
    cores = [1, 2, 4]
    cold = {p: statistics.median(run_privatized(p) for _ in range(REPS)) for p in cores}
    warm = {p: statistics.median(run_privatized_warm(p) for _ in range(REPS)) for p in cores}
    report("Handout timing (includes Pool start-up):", cold)
    report("Compute + reduction only (pool started before timing):", warm)
    with open("exp_c_results.json", "w") as f:
        json.dump({"with_pool_startup": cold, "compute_only": warm}, f, indent=2)
