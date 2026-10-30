# CS482 Lab Practicum - Memory Hierarchy, Contention and Scaling

**Student ID:** 230103182 | **N:** 1408 | **seed:** 73101 | **Host:** 13th Gen Intel(R) Core(TM) i9-13900HX | **Python:** 3.14.3

## Summary of bottleneck findings

On a 13th Gen Intel(R) Core(TM) i9-13900HX (24 physical cores / 32 threads, L3 36 MiB, 64-byte lines; salted N = 1408), every bottleneck we measured is a memory-system bottleneck rather than a lack of arithmetic. Walking a 128 MiB buffer, per-element latency rose from 0.64 ns at stride 1 to 3.06 ns at stride 8 (x4.8) - the point where every read needs its own 64-byte line - and stayed high for larger strides; the same effect made the i-k-j loop order 5.8x faster than i-j-k on the 704x704 block with bit-identical results. With numba workers, unpadded counters sharing one cache line ran 4.98x slower at P=8 than at P=1 (Karp-Flatt e(8) = 0.569), whereas 64-byte padding gave e(8) = 0.083 and privatization e(8) = 0.110. The padded e(2) = 0.42 is an outlier: the numba runs last only 12-20 ms, so OS scheduling noise is visible there, but the trend (e falls with P for padded, rises for unpadded) and the pure-Python run (unpadded e(8) = 0.95 vs padded 0.156) strongly indicate coherence (MESI) ping-pong as the cause of the slowdown, not the work itself. The 5-point stencil scaled best at P=4 (3.15x); beyond that the speedup fell back (2.37x at P=24, efficiency 10%). Fitting Amdahl's law gives an effective serial fraction s = 0.295 and a ceiling S_max = 3.4x. Weak scaling broke down as well: with the work per process held constant, time grew from 1.48 s at P=1 to 21.9 s at P=24 (scaled efficiency 0.07). Part of the drop beyond P=8 is also due to the hybrid CPU: P > 8 lands on the slower Gracemont E-cores. With an arithmetic intensity of only 0.1667 FLOP/B against a ridge point of 24.19 FLOP/B, the stencil is memory-bandwidth bound: the 'serial fraction' is really shared memory-bandwidth saturation, so more cores stop helping once DRAM/LLC bandwidth is used up.

## Reproduce

```
pip install numpy numba psutil
python src/profile_env.py --student-id 230103182
python src/stride_bench.py
python src/dense_kernel.py
python src/false_sharing.py
python src/scaling_study.py
python verify_outputs.py
```

Notes: shared counters use `multiprocessing.Array(..., lock=False)` (the default lock would serialize every
variant and hide false sharing); false-sharing speedups are work-normalised (`S_P = P*T1/TP`) because each worker
performs K increments; all timings exclude process start-up (workers start behind a Barrier).
