"""Milestone 0 - deterministic Student ID salting (exactly as specified in the handout)."""
import hashlib
import os


def get_student_params(student_id: str):
    h = int(hashlib.sha256(student_id.encode('utf-8')).hexdigest()[:8], 16)
    # Salted matrix dimension N between 1024 and 1536 (multiple of 64)
    N = 1024 + ((h % 9) * 64)
    # Salted stride benchmark buffer size: 2^24 elements (~134 MB double-precision)
    buffer_elements = 16_777_216
    # Seed for deterministic random number generation
    seed = h % 100_007
    return {"student_id": student_id, "N": N, "buffer_elements": buffer_elements, "seed": seed}


# ---------------------------------------------------------------------------
# Helpers shared by every milestone script
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load_params():
    """Read the salted parameters written by profile_env.py."""
    import json
    path = os.path.join(RESULTS, "salted_params.json")
    if not os.path.exists(path):
        raise SystemExit("results/salted_params.json not found - run "
                         "`python src/profile_env.py --student-id <ID>` first (Milestone 0).")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_profile():
    import json
    with open(os.path.join(RESULTS, "host_profile.json"), encoding="utf-8") as f:
        return json.load(f)
