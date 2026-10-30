"""Milestone 3 - strong / weak scaling of a 2-D 5-point Jacobi stencil + Roofline.

Kernel (one sweep, interior points only, boundaries fixed):
    new[i,j] = 0.25 * (old[i-1,j] + old[i+1,j] + old[i,j-1] + old[i,j+1])
Domain decomposition: P worker processes each own a contiguous block of rows.
Both grids live in multiprocessing.shared_memory (no pickling of data); workers
synchronize with a Barrier after every sweep (double buffering).
Every element is computed with exactly the same operation order for any P, so the
parallel result must be BITWISE identical to the P=1 reference.

Strong scaling : fixed N = N_salted, P = 1, 2, 4, ... , physical cores
Weak scaling   : N_P = ceil(N * sqrt(P))  ->  work(P) = O(P)

Usage: python src/scaling_study.py [--sweeps 100] [--reps 3] [--max-p 24]
                                   [--mem-bw-gbs 76.8] [--flops-per-cycle 16]
Output: results/scaling_metrics.tsv, results/roofline_summary.json,
        results/arrays/m3_checksums.json
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from multiprocessing import shared_memory

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import RESULTS, load_params, load_profile  # noqa: E402

FLOPS_PER_POINT = 4          # 3 additions + 1 multiplication
TERMINATE = -1


# ----------------------------------------------------------------------------- kernel
def sweep_rows(src, dst, r0, r1):
    """Update interior rows [r0, r1) of dst from src, allocation-free (out= ufuncs)."""
    d = dst[r0:r1, 1:-1]
    np.add(src[r0 - 1:r1 - 1, 1:-1], src[r0 + 1:r1 + 1, 1:-1], out=d)
    np.add(d, src[r0:r1, :-2], out=d)
    np.add(d, src[r0:r1, 2:], out=d)
    np.multiply(d, 0.25, out=d)


def initial_grid(N, seed):
    g = np.random.default_rng(seed).random((N, N))
    g[0, :] = 1.0          # hot top boundary, others random but fixed
    return g


def row_blocks(N, P):
    interior = N - 2
    base, extra = divmod(interior, P)
    out, r = [], 1
    for p in range(P):
        n = base + (1 if p < extra else 0)
        out.append((r, r + n))
        r += n
    return out


def serial_reference(N, seed, sweeps):
    a = initial_grid(N, seed)
    b = a.copy()
    for _ in range(sweeps):
        sweep_rows(a, b, 1, N - 1)
        a, b = b, a
    return a


def grid_digest(g):
    return hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- workers
def _worker(names, N, r0, r1, sweeps, barrier, stamps, wid):
    kw = {"track": False} if sys.version_info >= (3, 13) else {}   # parent owns/unlinks the segments
    shms = [shared_memory.SharedMemory(name=n, **kw) for n in names]
    grids = [np.ndarray((N, N), dtype=np.float64, buffer=s.buf) for s in shms]
    try:
        barrier.wait()
        t0 = time.perf_counter_ns()
        for t in range(sweeps):
            sweep_rows(grids[t % 2], grids[(t + 1) % 2], r0, r1)
            barrier.wait()                        # halo rows of neighbours are now final
        stamps[2 * wid] = t0
        stamps[2 * wid + 1] = time.perf_counter_ns()
    finally:
        del grids
        for s in shms:
            s.close()


def run_parallel(N, P, sweeps, seed):
    init = initial_grid(N, seed)
    shms = [shared_memory.SharedMemory(create=True, size=init.nbytes) for _ in range(2)]
    try:
        grids = [np.ndarray((N, N), dtype=np.float64, buffer=s.buf) for s in shms]
        grids[0][:] = init
        grids[1][:] = init                        # boundaries present in both buffers
        del init
        stamps = mp.Array("q", 2 * P, lock=False)
        barrier = mp.Barrier(P)
        procs = [mp.Process(target=_worker, args=([s.name for s in shms], N, r0, r1, sweeps, barrier, stamps, p))
                 for p, (r0, r1) in enumerate(row_blocks(N, P))]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
            if pr.exitcode != 0:
                raise RuntimeError(f"stencil worker failed (exit {pr.exitcode})")
        T = max(stamps[1::2]) - min(stamps[0::2])
        final = grids[sweeps % 2]
        checksum = float(final.sum())
        dig = grid_digest(final)
        del grids, final
        return T, checksum, dig
    finally:
        for s in shms:
            s.close()
            s.unlink()


# ----------------------------------------------------------------------------- analysis
def amdahl_lsq(points):
    """Least-squares serial fraction from (P, T_P/T_1) pairs:
       T_P/T_1 = s*(1-1/P) + 1/P   ->  s = sum(x*y)/sum(x*x)."""
    xs = [1 - 1 / P for P, _ in points if P > 1]
    ys = [r - 1 / P for P, r in points if P > 1]
    if not xs:
        return None
    return sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)


def p_list(max_p):
    ps, p = [], 1
    while p <= max_p:
        ps.append(p)
        p *= 2
    if ps[-1] != max_p:
        ps.append(max_p)
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-p", type=int, default=None, help="default: physical cores from host_profile.json")
    ap.add_argument("--mem-bw-gbs", type=float, default=None,
                    help="peak DRAM bandwidth GB/s (default: derived from DIMM speed x 2 channels x 8 B, else 76.8)")
    ap.add_argument("--flops-per-cycle", type=float, default=16.0,
                    help="DP FLOPs/cycle/core (AVX2: 2 FMA units x 4 doubles x 2 = 16)")
    ap.add_argument("--skip-weak", action="store_true")
    args = ap.parse_args()

    params, prof = load_params(), load_profile()
    N0, seed, sid = params["N"], params["seed"], params["student_id"]
    phys = prof["cpu"]["physical_cores"] or os.cpu_count()
    max_p = args.max_p or phys
    PS = p_list(max_p)
    print(f"N={N0}  sweeps={args.sweeps}  reps={args.reps}  P={PS}  (physical cores={phys})")

    # ---- serial single-process reference (verification baseline) ---------------
    ref = serial_reference(N0, seed, args.sweeps)
    ref_sum, ref_dig = float(ref.sum()), grid_digest(ref)
    del ref

    rows, checks = [], {"student_id": sid, "N": N0, "sweeps": args.sweeps, "seed": seed,
                        "reference_checksum": ref_sum, "reference_digest": ref_dig, "strong": {}, "weak": {}}

    def measure(N, P):
        ts, sums, digs = [], set(), set()
        for _ in range(args.reps):
            T, cs, dg = run_parallel(N, P, args.sweeps, seed)
            ts.append(T)
            sums.add(cs)
            digs.add(dg)
        return statistics.median(ts), ts, sums, digs

    # ---- strong scaling --------------------------------------------------------
    strong = {}
    for P in PS:
        T, ts, sums, digs = measure(N0, P)
        strong[P] = T
        match = digs == {ref_dig}
        checks["strong"][str(P)] = {"checksum": sorted(sums), "digest": sorted(digs), "bitwise_match_reference": match}
        print(f"[strong] P={P:3d} N={N0}: T={T / 1e9:.4f}s  match_ref={match}")
        rows.append({"mode": "strong", "P": P, "N": N0, "interior_points": (N0 - 2) ** 2, "times_ns": ts,
                     "T": T, "match": match})
    T1 = strong[1]
    s_fit = amdahl_lsq([(P, strong[P] / T1) for P in PS])
    for r in rows:
        P, T = r["P"], r["T"]
        S = T1 / T
        r["speedup"] = S
        r["efficiency_pct"] = 100 * S / P
        r["serial_fraction_s"] = ((1 / S) - (1 / P)) / (1 - 1 / P) if P > 1 else None
        r["amdahl_fit_S"] = 1 / (s_fit + (1 - s_fit) / P) if s_fit is not None else None

    # ---- weak scaling ----------------------------------------------------------
    weak_rows = []
    if not args.skip_weak:
        W1 = (N0 - 2) ** 2
        Tw1 = None
        for P in PS:
            NP = math.ceil(N0 * math.sqrt(P))
            T, ts, sums, digs = measure(NP, P)
            Tw1 = Tw1 or T
            WP = (NP - 2) ** 2
            if P == 1:
                match = digs == {ref_dig}
            else:  # independent serial reference for the bigger grid
                rg = serial_reference(NP, seed, args.sweeps)
                match = digs == {grid_digest(rg)}
                del rg
            checks["weak"][str(P)] = {"N": NP, "checksum": sorted(sums), "bitwise_match_reference": match}
            S_scaled = (P * W1 * Tw1) / (WP * T)
            weak_rows.append({"mode": "weak", "P": P, "N": NP, "interior_points": WP, "times_ns": ts, "T": T,
                              "match": match, "speedup": S_scaled, "efficiency_pct": 100 * S_scaled / P,
                              "serial_fraction_s": (P - S_scaled) / (P - 1) if P > 1 else None,
                              "amdahl_fit_S": (P - s_fit * (P - 1)) if s_fit is not None else None})
            print(f"[weak  ] P={P:3d} N={NP}: T={T / 1e9:.4f}s  scaled speedup={S_scaled:.3f}  match_ref={match}")
    rows += weak_rows

    # ---- write scaling_metrics.tsv --------------------------------------------
    cols = ["mode", "P", "N", "interior_points", "sweeps", "median_T_s", "speedup", "efficiency_pct",
            "serial_fraction_s", "model_prediction", "achieved_GFLOPs", "achieved_bandwidth_GBps_numpy_traffic",
            "bitwise_match_reference", "all_times_s"]
    BYTES_NUMPY = 104   # bytes moved per point by the 5-pass numpy implementation (see roofline JSON)
    with open(os.path.join(RESULTS, "scaling_metrics.tsv"), "w", encoding="utf-8") as f:
        f.write("# speedup: strong -> S_P = T1/TP ; weak -> Gustafson scaled S = P*W1*T1/(WP*TP)\n")
        f.write("# serial_fraction_s: strong -> Amdahl/Karp-Flatt s(P) ; weak -> Gustafson s = (P - S)/(P - 1)\n")
        f.write(f"# model_prediction: strong -> Amdahl S with least-squares s={s_fit} ; weak -> Gustafson P - s(P-1)\n")
        f.write("\t".join(cols) + "\n")
        for r in rows:
            pts = r["interior_points"] * args.sweeps
            vals = [r["mode"], r["P"], r["N"], r["interior_points"], args.sweeps, f"{r['T'] / 1e9:.6f}",
                    f"{r['speedup']:.4f}", f"{r['efficiency_pct']:.2f}",
                    "" if r["serial_fraction_s"] is None else f"{r['serial_fraction_s']:.5f}",
                    "" if r["amdahl_fit_S"] is None else f"{r['amdahl_fit_S']:.4f}",
                    f"{pts * FLOPS_PER_POINT / r['T']:.3f}", f"{pts * BYTES_NUMPY / r['T']:.3f}",
                    r["match"], ",".join(f"{t / 1e9:.6f}" for t in r["times_ns"])]
            f.write("\t".join(str(v) for v in vals) + "\n")
    os.makedirs(os.path.join(RESULTS, "arrays"), exist_ok=True)
    with open(os.path.join(RESULTS, "arrays", "m3_checksums.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    # ---- roofline --------------------------------------------------------------
    cache = prof["cache"]
    l3 = cache.get("L3", {}).get("size_bytes") or 0
    l2 = cache.get("L2", {}).get("size_bytes") or 0
    freq_ghz = (prof["cpu"]["base_frequency_mhz"] or 2000) / 1000
    bw, bw_src = args.mem_bw_gbs, "command line"
    if bw is None:
        mods = prof.get("memory", {}).get("memory_modules")
        speed = None
        if isinstance(mods, list) and mods:
            speed = max((m.get("ConfiguredClockSpeed") or m.get("Speed") or 0) for m in mods if isinstance(m, dict))
            channels = min(len(mods), 2)
        if speed:
            bw, bw_src = speed * 8 * channels / 1000, f"{channels} channel(s) x {speed} MT/s x 8 B (from DIMM info)"
        else:
            bw, bw_src = 76.8, "assumed dual-channel DDR5-4800 (2 x 4800 MT/s x 8 B); override with --mem-bw-gbs"
    peak = phys * freq_ghz * args.flops_per_cycle
    ridge = peak / bw
    ai_ideal_wa = FLOPS_PER_POINT / 24     # read old (8 B) + write new (8 B) + write-allocate (8 B)
    ai_ideal = FLOPS_PER_POINT / 16        # with non-temporal stores / no write-allocate
    ai_numpy = FLOPS_PER_POINT / BYTES_NUMPY
    ws_strong = 2 * N0 * N0 * 8
    best = max((r for r in rows if r["mode"] == "strong"), key=lambda r: r["speedup"])
    pts = best["interior_points"] * args.sweeps
    achieved = pts * FLOPS_PER_POINT / best["T"]
    roof = {
        "kernel": "2-D 5-point Jacobi stencil, float64, double-buffered",
        "flops_per_point": FLOPS_PER_POINT,
        "arithmetic_intensity": {
            "ideal_fused_kernel_with_write_allocate": {"bytes_per_point": 24, "AI_flop_per_byte": round(ai_ideal_wa, 4)},
            "ideal_fused_kernel_streaming_stores": {"bytes_per_point": 16, "AI_flop_per_byte": round(ai_ideal, 4)},
            "this_numpy_implementation": {
                "bytes_per_point": BYTES_NUMPY, "AI_flop_per_byte": round(ai_numpy, 4),
                "derivation": "5 ufunc passes over the block: add(up,down)->d = 8+8+8, add(d,left) = 8+8+8, "
                              "add(d,right) = 8+8+8, multiply(d,.25) = 8+8 -> 104 B/point (neighbour rows reuse "
                              "cache, write-allocate ignored)"},
        },
        "machine": {
            "cpu": prof["cpu"]["model_name"], "physical_cores": phys, "frequency_ghz_used": freq_ghz,
            "dp_flops_per_cycle_per_core": args.flops_per_cycle,
            "peak_compute_GFLOPs": round(peak, 1),
            "peak_dram_bandwidth_GBps": round(bw, 2), "bandwidth_source": bw_src,
            "ridge_point_flop_per_byte": round(ridge, 3),
            "L2_bytes": l2, "L3_bytes": l3,
        },
        "working_set": {
            "strong_scaling_bytes": ws_strong, "strong_scaling_MiB": round(ws_strong / 2**20, 1),
            "fits_in_L3": ws_strong <= l3,
            "weak_scaling_max_bytes": 2 * (math.ceil(N0 * math.sqrt(max_p)) ** 2) * 8,
            "note": ("Strong-scaling grids fit in L3, so traffic is served partly by the LLC rather than DRAM; "
                     "the kernel is still bandwidth-bound (LLC bandwidth instead of DRAM bandwidth)."
                     if ws_strong <= l3 else
                     "Strong-scaling grids exceed L3, so every sweep streams from DRAM."),
        },
        "attainable_GFLOPs_at_ideal_AI": round(min(peak, ai_ideal_wa * bw), 2),
        "attainable_GFLOPs_numpy_AI": round(min(peak, ai_numpy * bw), 2),
        "measured_best_strong": {"P": best["P"], "GFLOPs": round(achieved / 1, 3),
                                 "implied_bandwidth_GBps_numpy_traffic": round(pts * BYTES_NUMPY / best["T"], 2),
                                 "fraction_of_peak_compute_pct": round(100 * achieved / peak, 3)},
        "amdahl": {"least_squares_serial_fraction_s": s_fit,
                   "S_max_asymptotic": (1 / s_fit) if s_fit and s_fit > 0 else None},
        "regime": "MEMORY-BANDWIDTH BOUND" if ai_ideal_wa < ridge else "COMPUTE BOUND",
        "conclusion": (f"AI = {ai_ideal_wa:.3f} FLOP/B (ideal) and {ai_numpy:.3f} FLOP/B (numpy) are far below the "
                       f"ridge point {ridge:.2f} FLOP/B, so the stencil sits on the sloped (bandwidth) part of the "
                       f"roofline: adding cores helps only until the memory subsystem saturates, which is what "
                       f"limits the strong-scaling efficiency." if ai_ideal_wa < ridge else
                       "AI exceeds the ridge point: compute bound."),
    }
    with open(os.path.join(RESULTS, "roofline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(roof, f, indent=2)
    print(f"Amdahl least-squares s = {s_fit}  ->  S_max = {roof['amdahl']['S_max_asymptotic']}")
    print(f"Roofline: AI={ai_ideal_wa:.3f} (ideal) / {ai_numpy:.3f} (numpy) vs ridge {ridge:.2f} -> {roof['regime']}")
    print("Wrote results/scaling_metrics.tsv and results/roofline_summary.json")


if __name__ == "__main__":
    mp.freeze_support()
    main()
