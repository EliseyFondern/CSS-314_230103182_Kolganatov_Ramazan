"""Generate README.md (1-paragraph bottleneck summary + key numbers) from results/.
Run after all milestones:  python make_readme.py   - then read it and edit in your own words."""
import csv
import json
import os

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def j(n):
    with open(os.path.join(RES, n), encoding="utf-8") as f:
        return json.load(f)


def t(n):
    with open(os.path.join(RES, n), encoding="utf-8") as f:
        return list(csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"))


prof, par, li, roof = j("host_profile.json"), j("salted_params.json"), j("loop_interchange.json"), j("roofline_summary.json")
stride, fs, sc = t("stride_results.tsv"), t("false_sharing_karp_flatt.tsv"), t("scaling_metrics.tsv")
eng = stride[0]["engine"]
lat = {int(r["stride"]): float(r["latency_ns_per_element"]) for r in stride if r["engine"] == eng}
feng = fs[0]["engine"]


def fsv(var, P, key):
    return next(float(r[key]) for r in fs if r["engine"] == feng and r["variant"] == var and int(r["P"]) == P)


strong = [r for r in sc if r["mode"] == "strong"]
weak = [r for r in sc if r["mode"] == "weak"]
sP = strong[-1]
wP = weak[-1] if weak else None
cpu = prof["cpu"]
s_fit = roof["amdahl"]["least_squares_serial_fraction_s"]
smax = roof["amdahl"]["S_max_asymptotic"]

para = (
    f"On a {cpu['model_name']} ({cpu['physical_cores']} physical cores / {cpu['logical_threads']} threads, "
    f"L3 {prof['cache']['L3']['size_human']}, {prof['cache']['cache_line_bytes']}-byte lines; salted N = {par['N']}), "
    f"every bottleneck we measured is a memory-system bottleneck rather than a lack of arithmetic. "
    f"Walking a 128 MiB buffer, per-element latency rose from {lat[1]:.2f} ns at stride 1 to {lat[8]:.2f} ns at stride 8 "
    f"(x{lat[8] / lat[1]:.1f}) - the point where every read needs its own 64-byte line - and stayed high for larger strides; "
    f"the same effect made the i-k-j loop order {li['speedup_S']:.1f}x faster than i-j-k on the {li['block_M']}x{li['block_M']} block "
    f"with bit-identical results. With {feng} workers, unpadded counters sharing one cache line ran "
    f"{fsv('unpadded', 8, 'T_over_T1'):.2f}x slower at P=8 than at P=1 (Karp-Flatt e(8) = {fsv('unpadded', 8, 'karp_flatt_e'):.3f}), "
    f"whereas 64-byte padding gave e(8) = {fsv('padded', 8, 'karp_flatt_e'):.3f} and privatization e(8) = "
    f"{fsv('privatized', 8, 'karp_flatt_e'):.3f}, proving the slowdown was coherence (MESI) ping-pong, not the work itself. "
    f"The 5-point stencil reached a strong-scaling speedup of {float(sP['speedup']):.2f}x on P={sP['P']} "
    f"(efficiency {float(sP['efficiency_pct']):.0f}%); fitting Amdahl's law gives an effective serial fraction "
    f"s = {s_fit:.3f} and a ceiling S_max = {smax:.1f}x"
    + (f", while weak scaling kept a Gustafson scaled speedup of {float(wP['speedup']):.2f}x at P={wP['P']}" if wP else "")
    + f". With an arithmetic intensity of only {roof['arithmetic_intensity']['ideal_fused_kernel_with_write_allocate']['AI_flop_per_byte']} FLOP/B "
    f"against a ridge point of {roof['machine']['ridge_point_flop_per_byte']} FLOP/B, the stencil is {roof['regime'].lower()}: "
    f"the 'serial fraction' is really shared memory-bandwidth saturation, so more cores stop helping once DRAM/LLC bandwidth is used up."
)

md = f"""# CS482 Lab Practicum - Memory Hierarchy, Contention and Scaling

**Student ID:** {par['student_id']} | **N:** {par['N']} | **seed:** {par['seed']} | **Host:** {cpu['model_name']} | **Python:** {prof['python']['version']}

## Summary of bottleneck findings

{para}

## Reproduce

```
pip install numpy numba psutil
python src/profile_env.py --student-id {par['student_id']}
python src/stride_bench.py
python src/dense_kernel.py
python src/false_sharing.py
python src/scaling_study.py
python verify_outputs.py
```

Notes: shared counters use `multiprocessing.Array(..., lock=False)` (the default lock would serialize every
variant and hide false sharing); false-sharing speedups are work-normalised (`S_P = P*T1/TP`) because each worker
performs K increments; all timings exclude process start-up (workers start behind a Barrier).
"""
with open(os.path.join(os.path.dirname(RES), "README.md"), "w", encoding="utf-8") as f:
    f.write(md)
print(md)
