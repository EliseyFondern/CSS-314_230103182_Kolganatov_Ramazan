"""Part 3: Experiment C - Local Buffer Chunking.

Workers accumulate records in a private local buffer and flush one
pre-aggregated sum per 10,000 items, so the shared queue sees ~1,000 puts for
10,000,000 items instead of one put per item.

Pinned to the P-core hardware threads (logical CPUs 0-15) as in Experiment A;
median of REPS runs per P.
"""
import json
import statistics
import time
from multiprocessing import Barrier, Process, Queue

import psutil

P_CORES = list(range(16))
REPS = 3
CORES = [1, 2, 4, 8]


def buffered_worker(out_queue, items_per_worker, batch_size=10_000):
    buffer = []
    for i in range(items_per_worker):
        buffer.append(i % 7)
        if len(buffer) >= batch_size:
            out_queue.put(sum(buffer))  # Flushes pre-aggregated chunk once every 10,000 items
            buffer.clear()
    if buffer:
        out_queue.put(sum(buffer))
    out_queue.put(None)  # Sentinel end-of-work token


def run_buffered(p, total_items=10_000_000):
    q = Queue()
    items_per_worker = total_items // p
    workers = [
        Process(target=buffered_worker, args=(q, items_per_worker))
        for _ in range(p)
    ]
    t0 = time.perf_counter()
    for w in workers:
        w.start()
    total_sum, active = 0, p
    while active > 0:
        val = q.get()
        if val is None:
            active -= 1
        else:
            total_sum += val
    for w in workers:
        w.join()
    return time.perf_counter() - t0, total_sum


def ready_buffered_worker(out_queue, items_per_worker, start_barrier):
    start_barrier.wait()  # Interpreter is up; wait for the master to start the clock
    buffered_worker(out_queue, items_per_worker)


def run_buffered_ready(p, total_items=10_000_000):
    # Same work, but the clock starts only after every worker process has been
    # spawned and reached the barrier: on Windows each worker is a freshly
    # spawned interpreter, and that one-off start-up is not part of the algorithm.
    q = Queue()
    start_barrier = Barrier(p + 1)
    items_per_worker = total_items // p
    workers = [
        Process(target=ready_buffered_worker, args=(q, items_per_worker, start_barrier))
        for _ in range(p)
    ]
    for w in workers:
        w.start()
    start_barrier.wait()
    t0 = time.perf_counter()
    total_sum, active = 0, p
    while active > 0:
        val = q.get()
        if val is None:
            active -= 1
        else:
            total_sum += val
    t1 = time.perf_counter()
    for w in workers:
        w.join()
    return t1 - t0, total_sum


def expected_sum(p, total_items=10_000_000):
    n = total_items // p  # each worker emits i % 7 for i in range(n)
    return p * (21 * (n // 7) + sum(range(n % 7)))


def measure(fn):
    runtimes = {}
    for p in CORES:
        runs = [fn(p) for _ in range(REPS)]
        assert all(s == expected_sum(p) for _, s in runs), (p, runs)
        runtimes[p] = statistics.median(t for t, _ in runs)
    return runtimes


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
    handout = measure(run_buffered)
    ready = measure(run_buffered_ready)
    print("\n--- Buffered Queue Karp-Flatt Analysis ---")
    report("Handout timing (includes process start-up):", handout)
    report("Workers spawned before the clock starts:", ready)
    with open("exp_c_results.json", "w") as f:
        json.dump({"with_process_startup": handout, "workers_ready": ready}, f, indent=2)
