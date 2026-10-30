"""Milestone 4 - automated numerical verification.

NOTE: if the course staff distribute their own verify_outputs.py, use theirs
(drop it in the repository root, replacing this file). This is a compatible
self-check with the same checks described in Task 4.1.

Checks
  M0  host_profile.json fields present; salted_params.json == get_student_params(id)
  M1  stride checksums exact, stride cliff present; i-k-j matrix == i-j-k matrix within 1e-12,
      SHA-256 digest (values rounded to 6 decimals, salted with the student id) recomputed
  M2  every unpadded / padded / privatized run terminated with exactly P*K counts
  M3  every parallel stencil result bitwise equals the single-process reference
On success writes results/verification_receipt.sig
"""
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
from salt import get_student_params  # noqa: E402

failures = []


def check(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    if not ok:
        failures.append(name)


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def tsv(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"))


def salted_digest(arr, sid):
    h = hashlib.sha256(sid.encode())
    h.update(np.round(np.asarray(arr, dtype=np.float64), 6).tobytes())
    return h.hexdigest()


def main():
    # ---------------- M0
    try:
        prof, params = load("host_profile.json"), load("salted_params.json")
        sid = params["student_id"]
        need = [prof["cpu"]["model_name"], prof["cpu"]["physical_cores"], prof["cpu"]["logical_threads"],
                prof["cache"]["L1d"]["size_bytes"], prof["cache"]["L2"]["size_bytes"],
                prof["cache"]["L3"]["size_bytes"], prof["cache"]["cache_line_bytes"], prof["os"]["kernel"]]
        ok = all(v is not None for v in need) and params == get_student_params(sid)
        check("Milestone 0", ok, "Host topology and salted parameters validated." if ok
              else "missing telemetry or salted params do not match get_student_params()")
    except Exception as e:
        check("Milestone 0", False, f"{type(e).__name__}: {e}")
        sys.exit(1)

    # ---------------- M1
    try:
        rows = tsv("stride_results.tsv")
        exact = all(int(r["checksum"]) == int(r["elements_touched"]) for r in rows)
        eng = rows[0]["engine"]
        lat = {int(r["stride"]): float(r["latency_ns_per_element"]) for r in rows if r["engine"] == eng}
        strides_ok = sorted(lat) == [1, 2, 4, 8, 16, 32, 64, 128]
        cliff = lat[8] / lat[1]
        li = load("loop_interchange.json")
        z = np.load(os.path.join(RES, "arrays", "m1_matrices.npz"))
        diff = float(np.max(np.abs(z["C_ijk"] - z["C_ikj"])))
        d_ijk, d_ikj = salted_digest(z["C_ijk"], sid), salted_digest(z["C_ikj"], sid)
        ok = (exact and strides_ok and cliff > 1.5 and diff <= 1e-12 and d_ijk == d_ikj
              and d_ikj == li["verification"]["digest_ikj"] and li["N"] == params["N"])
        check("Milestone 1", ok, f"Matrix numerical digest matches reference: {d_ikj[:8]}... "
              f"(max|diff|={diff:.1e}, stride-8/stride-1 latency x{cliff:.2f}, S={li['speedup_S']:.2f})")
    except Exception as e:
        check("Milestone 1", False, f"{type(e).__name__}: {e}")

    # ---------------- M2
    try:
        c = load(os.path.join("arrays", "m2_counters.json"))
        runs = c["runs"]
        bad = [k for k, v in runs.items() if not v["exact"] or any(t != v["expected"] for t in v["totals"])]
        tsv_ok = all(r["final_count_exact"] == "True" for r in tsv("false_sharing_karp_flatt.tsv"))
        top = max(runs.values(), key=lambda v: v["expected"])
        ok = not bad and tsv_ok and c["student_id"] == sid
        check("Milestone 2", ok, f"Reduction sum exact match: {top['totals'][0]} == {top['expected']}."
              + (f" Mismatches: {bad}" if bad else f" ({len(runs)} configurations)"))
    except Exception as e:
        check("Milestone 2", False, f"{type(e).__name__}: {e}")

    # ---------------- M3
    try:
        m3 = load(os.path.join("arrays", "m3_checksums.json"))
        allruns = list(m3["strong"].items()) + list(m3["weak"].items())
        bad = [k for k, v in allruns if not v["bitwise_match_reference"]]
        sums_ok = all(abs(s - m3["reference_checksum"]) <= 1e-9 * abs(m3["reference_checksum"])
                      for v in m3["strong"].values() for s in v["checksum"])
        rows = tsv("scaling_metrics.tsv")
        ok = not bad and sums_ok and m3["N"] == params["N"] and all(r["bitwise_match_reference"] == "True" for r in rows)
        load("roofline_summary.json")
        check("Milestone 3", ok, "Stencil numerical tolerance verified "
              f"(reference checksum {m3['reference_checksum']:.6f}, {len(allruns)} parallel runs bitwise identical)."
              if ok else f"mismatch in {bad}")
    except Exception as e:
        check("Milestone 3", False, f"{type(e).__name__}: {e}")

    if failures:
        print(f"Verification FAILED: {failures}")
        sys.exit(1)

    files = sorted(f for f in os.listdir(RES) if os.path.isfile(os.path.join(RES, f)) and f != "verification_receipt.sig")
    hashes = {}
    for f in files:
        with open(os.path.join(RES, f), "rb") as fh:
            hashes[f] = hashlib.sha256(fh.read()).hexdigest()
    body = {"student_id": sid, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": prof["cpu"]["model_name"], "files_sha256": hashes}
    canon = json.dumps(body, sort_keys=True).encode()
    body["signature_sha256"] = hashlib.sha256(sid.encode() + canon).hexdigest()
    with open(os.path.join(RES, "verification_receipt.sig"), "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    print("All empirical datasets verified successfully. Emitting verification_receipt.sig.")


if __name__ == "__main__":
    main()
