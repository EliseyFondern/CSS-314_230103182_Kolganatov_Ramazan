# Practice Lab — Karp–Flatt Diagnostics

**Student:** Kolganatov Ramazan (230103182) | **Branch:** `practice-230103182`

## Hardware profile

| | |
|---|---|
| CPU | 13th Gen Intel Core i9-13900HX (Raptor Lake, hybrid) |
| Cores | 24 physical (8 P-cores with HT + 16 E-cores), `os.cpu_count()` = 32 |
| OS / Python | Windows 11 / CPython 3.14.3 (`multiprocessing` start method: spawn) |
| Pinning | All runs pinned to one hardware thread per P-core (logical CPUs 0,2,…,14); workers inherit the mask |
| Timing | Median of 3 runs per P |

Pinning was needed because without it Windows kept moving workers between P- and E-cores, and three
unpinned runs of Experiment A gave e(2) = 0.38 / 0.18 / 0.24, which was too noisy to diagnose anything.

## Results

| Experiment Mode | T1 (s) | T2 (s) | T4 (s) | e(2) | e(4) | Architectural Diagnosis |
|---|---|---|---|---|---|---|
| A: Serial Tail | 5.089 | 3.073 | 2.585 | 0.208 | 0.344 | Serial sleep + Turbo-Boost frequency drop (see below) |
| B: Lock Contention | 3.907 | 9.218 | 9.610 | 3.719 | 2.946 | Exploding e → mutex & cache-line thrashing (T4 = 2.46 × T1) |
| C: Privatized State (handout timing) | 0.897 | 0.467 | 0.383 | 0.041 | 0.237 | Contention gone; Pool start-up + Turbo dominate a 0.9 s job |
| C: Privatized State (compute only) | 0.511 | 0.334 | 0.210 | 0.306 | 0.214 | Contention gone; remaining loss is Turbo frequency drop |

Speedup / efficiency: A: S4 = 1.97×. B: S4 = 0.41× (negative scaling). C: S4 = 2.34× (E4 = 58 %) with the
handout timing, and 2.44× (E4 = 61 %) when the Pool is started before the clock.

## Rubric check

| Check | Result | Status |
|---|---|---|
| A: e(4)/e(2) ∈ [0.80, 1.25] | 0.344 / 0.208 = 1.65 | ✗ |
| B: T4 > T1 and e(4) ≫ e(2) | T4 = 9.61 s > T1 = 3.91 s ✓; e(4) = 2.95 < e(2) = 3.72 | partial |
| C: T4 < T1 and E4 > 70 % | T4 = 0.38 s < 0.90 s ✓; E4 = 58 % / 61 % | partial |

## Analysis

**Experiment A: why e(p) is not flat on this laptop.** I timed each 2,000,000-element `heavy_math` chunk inside
the workers. The same chunk took **0.268 s at P=1, 0.319 s at P=2 and 0.412 s at P=4**, even though starting
the Pool cost only 0.02–0.04 s. The work is identical, so each core simply runs slower when more cores are busy.
This is Intel Turbo Boost: one active core boosts close to 5.4 GHz, but with several active cores the laptop's
power and thermal limits pull the clock down. Karp–Flatt counts this lost per-core speed as "serial fraction",
so e(4) comes out above e(2). If I correct for the per-chunk slowdown (×1.19 at P=2, ×1.54 at P=4), the
remaining loss is the constant 0.40 s sleep, which is the flat Amdahl fraction the handout predicts.

**Experiment B: why e(4) < e(2) even though contention is extreme.** The `multiprocessing.Lock` is a kernel
semaphore on Windows. Two processes already saturate it, so T2 ≈ T4 ≈ 2.4 × T1. If T_p = c·T1 for every p ≥ 2,
the formula gives e(p) = (p·c − 1)/(p − 1). This **decreases** towards c as p grows: c = 2.4 gives
e(2) = 3.8 and e(4) = 2.9, which matches my measurements. So the signature here is not "e keeps growing" but
"e ≫ 1 and adding cores makes the program slower", and both conditions hold. In Experiment A every e stays
below 0.35; in B every e is above 2.9, about 10× larger.

**Experiment C: privatization works.** The 20,000,000 increments are 10× more work than in B, yet they finish
in 0.4–0.9 s instead of 4–10 s, the reduction is exact, and T4 < T1. Efficiency stays below 70 % for two
reasons. First, the handout timing includes starting the Pool, and on Windows each worker is a newly spawned
Python interpreter. Second, the same Turbo Boost drop measured in A applies: the ideal T4 = 0.511/4 = 0.128 s,
times the 1.54× per-core slowdown, gives 0.197 s, close to the measured 0.210 s. What is left is a hardware
clock effect, not a synchronization problem.

**Takeaway.** Karp–Flatt cleanly separated *contention* (B: e ≈ 3, negative speedup) from *no contention*
(A, C: e < 0.35). On a laptop CPU with an aggressive turbo, though, "e is constant" no longer reliably means
"fixed serial code": a falling clock frequency also makes e(p) grow. To isolate the algorithmic effect, the
experiment would need to run with Turbo Boost disabled (Windows power plan: maximum processor state 99 %).

## Reproduce

```
pip install psutil
cd PracticeLabHandout_230103182
python exp_a_serial.py
python exp_b_contention.py
python exp_c_fixed.py
```
