# Practice Lab (Variant B) — Granularity & IPC Contention

**Student:** Kolganatov Ramazan (230103182) | **Branch:** `variant-b-230103182`

## Hardware profile

| | |
|---|---|
| CPU | 13th Gen Intel Core i9-13900HX (Raptor Lake, hybrid) |
| Cores | 24 physical (8 P-cores with HT + 16 E-cores); `os.cpu_count()` = 32, `mp.cpu_count()` = 32 |
| OS / Python | Windows 11 / CPython 3.14.3 (`multiprocessing` start method: spawn) |
| Pinning | All runs pinned to the P-core hardware threads (logical CPUs 0–15); workers inherit the mask |
| Timing | Median of 3 runs per P |

## Results

| Configuration | T1 (s) | T2 (s) | T4 (s) | T8 (s) | S8 | e(2) | e(8) | Verdict |
|---|---|---|---|---|---|---|---|---|
| A: Micro (k=1) | 70.413 | 57.309 | 66.038 | 61.703 | 1.14 | 0.628 | 0.859 | Master dispatch serialization |
| A: Coarse (k=10000) | 8.738 | 6.166 | 3.888 | 3.257 | 2.68 | 0.411 | 0.283 | Dispatch amortized; scales |
| B: Queue Choke | 2.726 | 17.075 | 25.706 | 30.554 | 0.09 | 11.526 | 12.666 | IPC lock contention explosion |
| C: Buffered Batch (handout timing) | 0.679 | 0.475 | 0.409 | 0.459 | 1.48 | 0.400 | 0.630 | Contention gone; process spawn dominates |
| C: Buffered Batch (workers ready) | 0.633 | 0.434 | 0.246 | 0.160 | 3.97 | 0.371 | 0.145 | Amortized synchronization |

Additional Karp–Flatt values: A micro e(4) = 0.917; A coarse e(4) = 0.260; B e(4) = 12.239;
C handout e(4) = 0.471; C workers-ready e(4) = 0.185.
Both granularities in A return the same Collatz checksum (131,434,424), and every C run reproduces the exact
expected sum.

## Sign-off criteria

| Criterion | Result | Status |
|---|---|---|
| A: chunksize=1 suffers master dispatch starvation | k=1 takes 57–70 s for **every** P (S8 = 1.14); the same work with k=10000 takes 3.3–8.7 s. At P=8, micro-tasks are **19×** slower | ✓ |
| B: T8 > T1 and e(p) rises monotonically over {2, 4, 8} | T8 = 30.6 s vs T1 = 2.7 s (11× slower); e = 11.53 → 12.24 → 12.67 | ✓ |
| C: > 10× throughput increase | B at P=8 moves 13,092 items/s; C at P=8 moves 21.8 M items/s (handout timing), i.e. **~1,660×** | ✓ |
| C: S8 > 5.0 and e(8) ≤ 0.05 | S8 = 1.48 (handout timing) / 3.97 (workers ready); e(8) = 0.63 / 0.145 | ✗ |

## Analysis

**Experiment A: the granularity trap.** With `chunksize=1`, the master pickles and sends 1,000,000 separate
task envelopes and receives 1,000,000 results. Its own dispatch loop is the bottleneck, so runtime stays at
about 60 s whatever P is: adding workers only adds more mouths for one feeder to feed. The same Collatz work
in chunks of 10,000 (100 envelopes) is 8× faster already at P=1 and keeps improving up to P=8. Coarse scaling
stops at 2.7× for three reasons:
- Collatz cost grows with n, so the 100 static chunks are uneven, and the last chunks are the most expensive.
- Pool start-up (spawn) is included in the timing.
- The Turbo Boost clock falls as more cores become active.

**Experiment B: the queue choke-point.** Each `put()` takes the queue's inter-process lock, writes to an OS
pipe through a feeder thread, and wakes the reader. With one producer, that path runs uncontended at about
147,000 items/s. With 2, 4 and 8 producers the processes fight over the same lock and pipe, so throughput
falls to 23,400, 15,600 and 13,100 items/s. Runtime rises monotonically with P, and e(p) sits far above 1
and keeps growing (11.5 → 12.2 → 12.7). This is contention, not serial code: the algorithm has no serial
part at all.

**Experiment C: local buffer aggregation.** Each worker now sends one pre-summed value per 10,000 items, so
10,000,000 items need about 1,000 `put()` calls instead of 10,000,000. The job is 25× larger than B, yet it
finishes in 0.16–0.68 s instead of 2.7–30.6 s. Throughput rises about 1,660× at P=8, and the reduction is
exact. S8 still stays below 5 because the whole job is now tiny:
- With the handout timing, starting 8 fresh Python interpreters (Windows spawn) costs about as much as the
  work itself, so T8 does not drop below T4.
- When the clock starts only after all workers have reached a barrier, S8 = 3.97 and e(8) = 0.145.
- The gap left to S8 = 8 is the Turbo Boost frequency drop measured in the previous practice lab, where the
  same chunk ran 1.54× slower with 4 active cores than with 1, plus HT siblings sharing a core at P=8.

**Takeaway.** Karp–Flatt separates the three regimes clearly:
- **dispatch-bound:** e ≈ 0.6–0.9 and no speedup at all;
- **IPC-contention-bound:** e > 11 with negative scaling;
- **amortized:** e falls as P grows (0.37 → 0.19 → 0.15).

On this laptop the remaining e in C comes from hardware clock behaviour and process start-up, not from
synchronization.

## Reproduce

```
pip install psutil
cd CS482_VariantB_230103182
python exp_a_granularity.py
python exp_b_queue.py
python exp_c_buffered.py
```
