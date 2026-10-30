"""Milestone 2 - false-sharing pathology, remediation, Karp-Flatt diagnosis.

Variants (each worker p performs K increments, so a run with P workers does P*K total):
  unpadded   : counters at shared_arr[p]      -> 8 counters share ONE 64-byte line
  padded     : counters at shared_arr[p * 8]  -> every counter on its own 64-byte line
  privatized : worker counts in a private local, then ONE write to shared_arr[p]

Engines:
  python : the handout's literal code (`counter_array[worker_id] += 1` on a
           multiprocessing.Array).  The CPython interpreter spends ~50-100 ns per
           iteration, which dilutes - but does not hide - the coherence penalty.
  numba  : the same loop compiled to machine code over a NumPy view of the SAME
           multiprocessing.Array buffer.  The index is re-read from memory every
           iteration (possible aliasing), so the compiler cannot keep the counter
           in a register - every iteration is a real load+store to the shared line,
           which exposes the MESI ping-pong at full strength.

IMPORTANT: the arrays are created with lock=False.  With the default lock=True every
`arr[i] += 1` takes a process-shared mutex, so ALL variants would serialize on the
lock and you would be measuring lock contention, not false sharing.

Timing: workers start together behind a Barrier and record perf_counter_ns() start/end
in a shared array, so process spawn / import time is excluded.
T(P) = max(end) - min(start).

Karp-Flatt: total work grows with P (P*K increments), so speedup is work-normalised:
    S_P = (P*K / T_P) / (K / T_1) = P*T_1/T_P
    e(P) = (1/S_P - 1/P) / (1 - 1/P)          (handout formula, applied to this S_P)
Perfect scaling (T_P == T_1) gives e = 0; e -> 1 means fully serialized.

Usage: python src/false_sharing.py [--engines numba,python] [--K 5000000] [--reps 3]
Output: results/false_sharing_karp_flatt.tsv (+ results/arrays/m2_counters.json)
"""
import argparse
import ctypes
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import RESULTS, load_params  # noqa: E402

PS = [1, 2, 4, 8]
PAD = 8          # 8 x int64 = 64 bytes
VARIANTS = ["unpadded", "padded", "privatized"]

try:
    import numpy as np
    from numba import njit

    @njit(cache=True)
    def _nb_shared_inc(arr, ix, iterations):
        # arr[ix[0]] : ix[0] must be reloaded after every store (arr may alias ix),
        # so the increment is a genuine memory read-modify-write each iteration.
        for _ in range(iterations):
            arr[ix[0]] += 1

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


# ----------------------------------------------------------------------------- workers
def worker_unpadded(counter_array, worker_id, iterations):
    # Pathological: counter_array is a contiguous array of 64-bit integers.
    # worker 0 writes to index 0; worker 1 writes to index 1.
    # Indices 0 through 7 reside on the EXACT SAME 64-byte physical cache line!
    for _ in range(iterations):
        counter_array[worker_id] += 1


def worker_padded(counter_array, worker_id, iterations):
    idx = worker_id * PAD                       # each counter on its own cache line
    for _ in range(iterations):
        counter_array[idx] += 1


def worker_privatized(counter_array, worker_id, iterations):
    local = 0                                   # private stack/register variable
    for _ in range(iterations):
        local += 1
    counter_array[worker_id] = local            # single write at the end


def _numba_kernel(variant, shared, worker_id, iterations):
    view = np.frombuffer(shared, dtype=np.int64)
    if variant == "unpadded":
        _nb_shared_inc(view, np.array([worker_id], dtype=np.int64), iterations)
    elif variant == "padded":
        _nb_shared_inc(view, np.array([worker_id * PAD], dtype=np.int64), iterations)
    else:  # privatized: identical loop, but on a private process-local buffer
        local = np.zeros(PAD, dtype=np.int64)
        _nb_shared_inc(local, np.array([0], dtype=np.int64), iterations)
        view[worker_id] = local[0]


def _entry(engine, variant, shared, worker_id, iterations, barrier, stamps):
    if engine == "numba":                        # JIT warm-up outside the timed region
        _nb_shared_inc(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), 1)
    barrier.wait()
    t0 = time.perf_counter_ns()
    if engine == "python":
        {"unpadded": worker_unpadded, "padded": worker_padded,
         "privatized": worker_privatized}[variant](shared, worker_id, iterations)
    else:
        _numba_kernel(variant, shared, worker_id, iterations)
    t1 = time.perf_counter_ns()
    stamps[2 * worker_id] = t0
    stamps[2 * worker_id + 1] = t1


def run_once(engine, variant, P, K):
    size = 8 if variant != "padded" else 8 * PAD
    shared_arr = mp.Array(ctypes.c_int64, size, lock=False)   # no lock -> pure coherence traffic
    stamps = mp.Array(ctypes.c_int64, 2 * P, lock=False)
    barrier = mp.Barrier(P)
    procs = [mp.Process(target=_entry, args=(engine, variant, shared_arr, p, K, barrier, stamps))
             for p in range(P)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"worker failed (exit {pr.exitcode})")
    starts, ends = stamps[0::2], stamps[1::2]
    total = sum(shared_arr[:])
    return max(ends) - min(starts), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=None, help="numba,python (default: both if numba is installed)")
    ap.add_argument("--K", type=int, default=5_000_000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--start-method", default=None, help="spawn/fork/forkserver (Windows always uses spawn)")
    args = ap.parse_args()
    if args.start_method:
        mp.set_start_method(args.start_method, force=True)
    engines = args.engines.split(",") if args.engines else (["numba", "python"] if HAVE_NUMBA else ["python"])
    if "numba" in engines and not HAVE_NUMBA:
        sys.exit("numba not installed (pip install numba) or use --engines python")

    params = load_params()
    K = args.K
    print(f"K={K:,} increments per worker, P in {PS}, reps={args.reps}, logical CPUs={os.cpu_count()}")
    if os.cpu_count() < max(PS):
        print(f"WARNING: only {os.cpu_count()} logical CPUs - P > {os.cpu_count()} is oversubscribed.")

    rows, counters = [], {}
    for eng in engines:
        for var in VARIANTS:
            T1 = None
            for P in PS:
                times, totals = [], []
                for _ in range(args.reps):
                    t, tot = run_once(eng, var, P, K)
                    times.append(t)
                    totals.append(tot)
                T = statistics.median(times)
                T1 = T1 or T
                S = P * T1 / T                        # work-normalised speedup
                e = ((1 / S) - (1 / P)) / (1 - 1 / P) if P > 1 else None
                ok = all(x == P * K for x in totals)
                counters[f"{eng}/{var}/P{P}"] = {"totals": totals, "expected": P * K, "exact": ok}
                rows.append({
                    "engine": eng, "variant": var, "P": P, "K_per_worker": K, "total_increments": P * K,
                    "median_T_s": round(T / 1e9, 6), "T_over_T1": round(T / T1, 4),
                    "ns_per_increment_per_worker": round(T / K, 3),
                    "throughput_Minc_per_s": round(P * K / T * 1e3, 2),
                    "speedup_S_P": round(S, 4), "efficiency_pct": round(100 * S / P, 2),
                    "karp_flatt_e": "" if e is None else round(e, 5),
                    "final_count_exact": ok, "all_times_s": ",".join(f"{x / 1e9:.6f}" for x in times),
                })
                print(f"[{eng:6}] {var:10} P={P}: T={T / 1e9:8.4f}s  T/T1={T / T1:6.2f}  "
                      f"S={S:6.2f}  e={'-' if e is None else f'{e:.4f}'}  count_ok={ok}")

    cols = list(rows[0].keys())
    with open(os.path.join(RESULTS, "false_sharing_karp_flatt.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    os.makedirs(os.path.join(RESULTS, "arrays"), exist_ok=True)
    with open(os.path.join(RESULTS, "arrays", "m2_counters.json"), "w", encoding="utf-8") as f:
        json.dump({"student_id": params["student_id"], "K": K, "runs": counters}, f, indent=2)

    print("\nKarp-Flatt summary  e(P) for P = 2, 4, 8")
    for eng in engines:
        for var in VARIANTS:
            es = [r["karp_flatt_e"] for r in rows if r["engine"] == eng and r["variant"] == var and r["P"] > 1]
            print(f"  {eng:6} {var:10}: " + "  ".join(f"{x:.4f}" for x in es))
    print("Wrote results/false_sharing_karp_flatt.tsv")


if __name__ == "__main__":
    mp.freeze_support()
    main()
