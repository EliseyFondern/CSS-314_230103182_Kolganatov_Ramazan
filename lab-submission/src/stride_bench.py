"""Milestone 1, Task 1.1 - stride latency cliff benchmark.

A flat 2^24-element float64 buffer (128 MiB, larger than L3) is traversed with
stride s in {1,2,4,8,16,32,64,128}; total += data[i]. 5 warm-up + 5 recorded
iterations per stride, median time is logged.

Engines:
  numba  - compiled loop (the real hardware signal; interpreter overhead removed)
  numpy  - np.add.reduce over a strided view (fallback if numba is missing)
  python - pure interpreted loop (only for comparison: ~30-60 ns/element of
           interpreter overhead hides most of the memory-system effect)

Usage: python src/stride_bench.py [--engines numba,numpy]
Output: results/stride_results.tsv
"""
import argparse
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import RESULTS, load_params  # noqa: E402

STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]
WARMUP, RECORDED = 5, 5
LINE = 64          # bytes per cache line
ELEM = 8           # bytes per float64

try:
    from numba import njit

    @njit(cache=True)
    def _strided_sum_numba(a, s):
        # 4 independent accumulators so the stride-1 case is limited by memory,
        # not by the 4-cycle FP-add dependency chain.
        n = a.shape[0]
        t0 = 0.0
        t1 = 0.0
        t2 = 0.0
        t3 = 0.0
        i = 0
        s4 = 4 * s
        while i + 3 * s < n:
            t0 += a[i]
            t1 += a[i + s]
            t2 += a[i + 2 * s]
            t3 += a[i + 3 * s]
            i += s4
        while i < n:
            t0 += a[i]
            i += s
        return t0 + t1 + t2 + t3
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def strided_sum_numpy(a, s):
    return float(np.add.reduce(a[::s]))


def strided_sum_python(a, s):
    total = 0.0
    lst = a  # index the ndarray directly
    for i in range(0, len(lst), s):
        total += lst[i]
    return total


def bench(fn, data, s, warmup, recorded):
    for _ in range(warmup):
        fn(data, s)
    times, total = [], None
    for _ in range(recorded):
        t0 = time.perf_counter_ns()
        total = fn(data, s)
        times.append(time.perf_counter_ns() - t0)
    return times, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=None,
                    help="comma list of numba,numpy,python (default: numba,numpy or numpy if numba missing)")
    ap.add_argument("--python-elements", type=int, default=2_097_152,
                    help="buffer size used by the slow pure-python engine (default 2^21)")
    args = ap.parse_args()

    params = load_params()
    n = params["buffer_elements"]
    engines = args.engines.split(",") if args.engines else (["numba", "numpy"] if HAVE_NUMBA else ["numpy"])
    if "numba" in engines and not HAVE_NUMBA:
        sys.exit("numba not installed: pip install numba   (or use --engines numpy)")

    # Allocate flat array in physical memory (np.ones touches every page -> no lazy page faults)
    data = np.ones(n, dtype=np.float64)
    print(f"Buffer: {n} float64 = {n * ELEM / 2**20:.0f} MiB")

    rows = []
    for eng in engines:
        fn = {"numba": _strided_sum_numba if HAVE_NUMBA else None,
              "numpy": strided_sum_numpy, "python": strided_sum_python}[eng]
        buf = data if eng != "python" else data[:args.python_elements]
        w, r = (WARMUP, RECORDED) if eng != "python" else (1, 3)
        base = None
        for s in STRIDES:
            times, total = bench(fn, buf, s, w, r)
            touched = math.ceil(len(buf) / s)
            assert int(total) == touched, f"checksum mismatch {total} != {touched}"
            med = statistics.median(times)
            lat = med / touched
            lines = math.ceil(len(buf) * ELEM / LINE) if s * ELEM < LINE else touched
            useful = min(s * ELEM, LINE)  # bytes of DRAM traffic per useful 8-byte element
            base = base or lat
            rows.append({
                "engine": eng, "stride": s, "stride_bytes": s * ELEM, "buffer_elements": len(buf),
                "elements_touched": touched, "cache_lines_touched": lines,
                "bytes_moved_per_element": useful,
                "median_time_ns": int(med), "min_time_ns": min(times), "max_time_ns": max(times),
                "latency_ns_per_element": round(lat, 4),
                "slowdown_vs_stride1": round(lat / base, 3),
                "effective_bandwidth_GBps": round(lines * LINE / med, 3),
                "useful_bandwidth_GBps": round(touched * ELEM / med, 3),
                "checksum": int(total),
                "all_times_ns": ",".join(str(t) for t in times),
            })
            print(f"[{eng:6}] stride {s:4d}: {lat:8.3f} ns/elem  x{lat / base:6.2f}  "
                  f"line-BW {lines * LINE / med:6.2f} GB/s")

    cols = list(rows[0].keys())
    with open(os.path.join(RESULTS, "stride_results.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print("Wrote results/stride_results.tsv")


if __name__ == "__main__":
    main()
