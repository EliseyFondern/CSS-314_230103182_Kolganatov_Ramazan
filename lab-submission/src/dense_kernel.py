"""Milestone 1, Task 1.2 - loop interchange on a dense matmul kernel (i-j-k vs i-k-j).

Runs on a sub-block M = N/2 of the salted dimension N.  The kernels are compiled
with Numba (allowed by the handout: "compile via Cython/Numba/ctypes") so that
the measured difference is the memory-access pattern, not interpreter overhead.
If Numba is missing, a pure-Python flat-list version is used on a smaller block
(--block), which the JSON records honestly.

Usage: python src/dense_kernel.py [--runs 3] [--engine numba|python] [--block M]
Output: results/loop_interchange.json  (+ results/arrays/m1_matrices.npz for verification)
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import RESULTS, load_params  # noqa: E402

try:
    from numba import njit

    @njit(cache=True)
    def matmul_ijk(A, B, C):
        M = A.shape[0]
        for i in range(M):
            for j in range(M):
                acc = C[i, j]
                for k in range(M):
                    acc += A[i, k] * B[k, j]      # B[k, j]: stride-M walk down a column
                C[i, j] = acc

    @njit(cache=True)
    def matmul_ikj(A, B, C):
        M = A.shape[0]
        for i in range(M):
            for k in range(M):
                a = A[i, k]                       # hoisted into a register
                for j in range(M):
                    C[i, j] += a * B[k, j]        # stride-1 over rows of B and C
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


# ---- pure-Python fallback on flat 1-D lists -------------------------------------------
def py_matmul_ijk(A, B, C, M):
    for i in range(M):
        iM = i * M
        for j in range(M):
            acc = C[iM + j]
            for k in range(M):
                acc += A[iM + k] * B[k * M + j]
            C[iM + j] = acc


def py_matmul_ikj(A, B, C, M):
    for i in range(M):
        iM = i * M
        for k in range(M):
            a = A[iM + k]
            kM = k * M
            for j in range(M):
                C[iM + j] += a * B[kM + j]


def digest(arr, student_id):
    h = hashlib.sha256(student_id.encode())
    h.update(np.round(np.asarray(arr, dtype=np.float64), 6).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--engine", choices=["numba", "python"], default="numba" if HAVE_NUMBA else "python")
    ap.add_argument("--block", type=int, default=None, help="override block size M (default N/2)")
    args = ap.parse_args()
    if args.engine == "numba" and not HAVE_NUMBA:
        sys.exit("numba not installed: pip install numba   (or --engine python --block 128)")

    p = load_params()
    N, seed = p["N"], p["seed"]
    M = args.block or N // 2
    rng = np.random.default_rng(seed)
    A = rng.random((M, M))
    B = rng.random((M, M))
    print(f"N={N}  block M={M}  engine={args.engine}  runs={args.runs}")

    def run(kind):
        if args.engine == "numba":
            C = np.zeros((M, M))
            f = matmul_ijk if kind == "ijk" else matmul_ikj
            t0 = time.perf_counter_ns()
            f(A, B, C)
            return time.perf_counter_ns() - t0, C
        Af, Bf, Cf = A.ravel().tolist(), B.ravel().tolist(), [0.0] * (M * M)
        f = py_matmul_ijk if kind == "ijk" else py_matmul_ikj
        t0 = time.perf_counter_ns()
        f(Af, Bf, Cf, M)
        return time.perf_counter_ns() - t0, np.array(Cf).reshape(M, M)

    if args.engine == "numba":   # JIT compile outside the timed region
        tiny = np.ones((4, 4))
        matmul_ijk(tiny, tiny, np.zeros((4, 4)))
        matmul_ikj(tiny, tiny, np.zeros((4, 4)))

    res = {}
    for kind in ("ijk", "ikj"):
        ts = []
        for r in range(args.runs):
            t, C = run(kind)
            ts.append(t)
            print(f"  {kind} run {r + 1}: {t / 1e9:.4f} s")
        res[kind] = (ts, C)

    (t_ijk, C_ijk), (t_ikj, C_ikj) = res["ijk"], res["ikj"]
    med_ijk, med_ikj = statistics.median(t_ijk), statistics.median(t_ikj)
    max_abs = float(np.max(np.abs(C_ijk - C_ikj)))
    ref = A @ B
    rel_vs_blas = float(np.max(np.abs(C_ikj - ref)) / np.max(np.abs(ref)))
    flops = 2 * M ** 3
    out = {
        "student_id": p["student_id"], "N": N, "block_M": M, "engine": args.engine, "runs": args.runs,
        "naive_ijk": {"times_ns": t_ijk, "median_s": med_ijk / 1e9, "GFLOPs": flops / med_ijk},
        "interchanged_ikj": {"times_ns": t_ikj, "median_s": med_ikj / 1e9, "GFLOPs": flops / med_ikj},
        "speedup_S": med_ijk / med_ikj,
        "verification": {
            "max_abs_diff_ijk_vs_ikj": max_abs, "within_1e-12": max_abs <= 1e-12,
            "max_rel_diff_vs_numpy_matmul": rel_vs_blas,
            "digest_ijk": digest(C_ijk, p["student_id"]), "digest_ikj": digest(C_ikj, p["student_id"]),
        },
        "explanation": ("i-j-k reads B[k,j] down a column: consecutive k are M*8 bytes apart, so every "
                        "inner-loop iteration touches a new cache line and only 8 of 64 bytes are used. "
                        "i-k-j hoists A[i,k] and streams rows of B and C with stride 1, so each 64-byte "
                        "line serves 8 doubles, the hardware prefetcher is effective and the loop vectorizes."),
    }
    os.makedirs(os.path.join(RESULTS, "arrays"), exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, "arrays", "m1_matrices.npz"), C_ijk=C_ijk, C_ikj=C_ikj)
    with open(os.path.join(RESULTS, "loop_interchange.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"median ijk {med_ijk / 1e9:.4f} s | ikj {med_ikj / 1e9:.4f} s | speedup S = {out['speedup_S']:.2f}x")
    print(f"max |C_ijk - C_ikj| = {max_abs:.3e}   (<=1e-12: {max_abs <= 1e-12})")
    print("Wrote results/loop_interchange.json")


if __name__ == "__main__":
    main()
