"""Part 2: Experiment B - The Inter-Process Queue Choke-Point.

Every worker iteration does one put() on a shared multiprocessing.Queue, so
every item crosses the IPC boundary and the master drains them one by one.

Pinned to the P-core hardware threads (logical CPUs 0-15) as in Experiment A;
median of REPS runs per P.
"""
import json
import statistics
import time
from multiprocessing import Process, Queue

import psutil

P_CORES = list(range(16))
REPS = 3


def queue_worker(out_queue, items_per_worker):
    for i in range(items_per_worker):
        # Pathological pattern: Every single iteration crosses an IPC boundary
        out_queue.put(i % 7)


def run_queue_benchmark(p, total_items=400_000):
    q = Queue()
    items_per_worker = total_items // p
    workers = [
        Process(target=queue_worker, args=(q, items_per_worker))
        for _ in range(p)
    ]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    # Master consumes all items from the shared channel
    collected = 0
    while collected < total_items:
        _ = q.get()
        collected += 1
    for w in workers:
        w.join()
    t1 = time.perf_counter()
    return t1 - t0


if __name__ == "__main__":
    psutil.Process().cpu_affinity(P_CORES)
    cores = [1, 2, 4, 8]
    runtimes = {p: statistics.median(run_queue_benchmark(p) for _ in range(REPS)) for p in cores}
    t1 = runtimes[1]
    print("\n--- Queue Contention Karp-Flatt Analysis ---")
    print(f"P= 1 | Time={t1:6.3f}s")
    for p in [2, 4, 8]:
        tp = runtimes[p]
        sp = t1 / tp
        e = ((p * tp / t1) - 1.0) / (p - 1.0)
        print(f"P={p:2d} | Time={tp:6.3f}s | Speedup={sp:5.2f}x | Karp-Flatt e={e:6.3f}")
    with open("exp_b_results.json", "w") as f:
        json.dump(runtimes, f, indent=2)
