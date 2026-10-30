"""Milestone 0 - host topology profiling + salted parameter generation.

Usage:  python src/profile_env.py --student-id 12345678

Writes results/host_profile.json and results/salted_params.json.
Works on Windows (registry + GetLogicalProcessorInformation via ctypes),
Linux (/proc/cpuinfo + /sys/devices/system/cpu) and macOS (sysctl).
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from salt import RESULTS, get_student_params  # noqa: E402


# ---------------------------------------------------------------------------
# Microarchitecture lookup (best effort, from the marketing name / CPUID model)
# ---------------------------------------------------------------------------
NAME_TO_UARCH = [
    (r"14th Gen Intel", "Raptor Lake Refresh (Raptor Cove P-cores + Gracemont E-cores)"),
    (r"13th Gen Intel", "Raptor Lake (Raptor Cove P-cores + Gracemont E-cores)"),
    (r"12th Gen Intel", "Alder Lake (Golden Cove P-cores + Gracemont E-cores)"),
    (r"11th Gen Intel", "Tiger Lake / Rocket Lake (Willow/Cypress Cove)"),
    (r"Core\(TM\) Ultra .* 2\d\dV", "Lunar Lake (Lion Cove + Skymont)"),
    (r"Core\(TM\) Ultra .* 2\d\d", "Arrow Lake (Lion Cove + Skymont)"),
    (r"Core\(TM\) Ultra", "Meteor Lake (Redwood Cove + Crestmont)"),
    (r"Ryzen .* 9\d\d\d", "Zen 5"),
    (r"Ryzen .* [78]\d\d\d", "Zen 4"),
    (r"Ryzen .* 5\d\d\d", "Zen 3"),
    (r"Apple M4", "Apple M4"), (r"Apple M3", "Apple M3"),
    (r"Apple M2", "Apple M2"), (r"Apple M1", "Apple M1"),
]
INTEL_FAM6_MODELS = {
    0x8F: "Sapphire Rapids", 0xCF: "Emerald Rapids", 0xAD: "Granite Rapids",
    0x6A: "Ice Lake-SP", 0x55: "Skylake-SP / Cascade Lake", 0xB7: "Raptor Lake",
    0xBA: "Raptor Lake", 0xBF: "Raptor Lake", 0x97: "Alder Lake", 0x9A: "Alder Lake",
    0xAA: "Meteor Lake", 0xC6: "Arrow Lake", 0xBD: "Lunar Lake", 0x8C: "Tiger Lake",
    0xA7: "Rocket Lake", 0x9E: "Coffee Lake", 0x8E: "Kaby/Whiskey/Comet Lake",
}


def guess_uarch(name, family=None, model=None):
    for pat, arch in NAME_TO_UARCH:
        if re.search(pat, name or ""):
            return arch
    if family == 6 and model in INTEL_FAM6_MODELS:
        return INTEL_FAM6_MODELS[model]
    return "unknown"


def fmt_bytes(n):
    if n is None:
        return None
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:g} {unit}" if unit == "B" else f"{n:.0f} {unit}" if n == int(n) else f"{n:.1f} {unit}"
        n /= 1024


def summarize_caches(instances):
    """instances: list of dicts {level, type, size, line, sharing}.
    Returns per-level summary (distinct sizes with instance counts) - important on
    hybrid CPUs where P-cores and E-cores have different L1/L2 sizes."""
    out = {}
    for lvl, typ, key in ((1, "Data", "L1d"), (1, "Instruction", "L1i"), (2, "Unified", "L2"), (3, "Unified", "L3")):
        sel = [c for c in instances if c["level"] == lvl and (c["type"] == typ or (lvl > 1))]
        if not sel:
            continue
        variants = {}
        for c in sel:
            k = (c["size"], c.get("sharing"))
            variants[k] = variants.get(k, 0) + 1
        out[key] = {
            "size_bytes": max(c["size"] for c in sel),  # largest variant (used by later milestones)
            "size_human": fmt_bytes(max(c["size"] for c in sel)),
            "instances": len(sel),
            "total_bytes": sum(c["size"] for c in sel),
            "variants": [{"size_bytes": s, "size_human": fmt_bytes(s), "logical_cpus_sharing": sh, "count": n}
                         for (s, sh), n in sorted(variants.items(), key=lambda kv: -kv[1])],
        }
    lines = {c["line"] for c in instances if c.get("line")}
    out["cache_line_bytes"] = max(lines) if lines else 64
    return out


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def probe_windows():
    import ctypes
    from ctypes import wintypes
    import winreg

    info = {}
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    info["model_name"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    info["base_frequency_mhz"] = int(winreg.QueryValueEx(key, "~MHz")[0])
    info["identifier"] = winreg.QueryValueEx(key, "Identifier")[0]
    m = re.search(r"Family (\d+) Model (\d+)", info["identifier"])
    fam, mod = (int(m.group(1)), int(m.group(2))) if m else (None, None)

    class CACHE_DESCRIPTOR(ctypes.Structure):
        _fields_ = [("Level", ctypes.c_ubyte), ("Associativity", ctypes.c_ubyte),
                    ("LineSize", wintypes.WORD), ("Size", wintypes.DWORD), ("Type", ctypes.c_int)]

    class _U(ctypes.Union):
        _fields_ = [("Cache", CACHE_DESCRIPTOR), ("Reserved", ctypes.c_ulonglong * 2)]

    class SLPI(ctypes.Structure):
        _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_int), ("u", _U)]

    k32 = ctypes.windll.kernel32
    length = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformation(None, ctypes.byref(length))
    buf = (SLPI * (length.value // ctypes.sizeof(SLPI)))()
    if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(length)):
        raise OSError("GetLogicalProcessorInformation failed")
    cores, smt_cores, caches = 0, 0, []
    ctype_names = {0: "Unified", 1: "Instruction", 2: "Data", 3: "Trace"}
    for e in buf:
        if e.Relationship == 0:  # RelationProcessorCore
            cores += 1
            if bin(e.ProcessorMask).count("1") > 1:
                smt_cores += 1
        elif e.Relationship == 2:  # RelationCache
            c = e.u.Cache
            caches.append({"level": c.Level, "type": ctype_names.get(c.Type, str(c.Type)),
                           "size": c.Size, "line": c.LineSize,
                           "sharing": bin(e.ProcessorMask).count("1")})
    info["physical_cores"] = cores
    info["cores_with_smt"] = smt_cores
    info["logical_threads"] = os.cpu_count()
    info["caches_raw"] = caches
    info["cpuid_family"], info["cpuid_model"] = fam, mod

    # Optional: installed DRAM speed (for the roofline bandwidth estimate)
    try:
        ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "Get-CimInstance Win32_PhysicalMemory | Select-Object Speed,ConfiguredClockSpeed,Capacity,SMBIOSMemoryType | ConvertTo-Json"],
                            capture_output=True, text=True, timeout=20)
        dimms = json.loads(ps.stdout)
        info["memory_modules"] = dimms if isinstance(dimms, list) else [dimms]
    except Exception as exc:  # pragma: no cover
        info["memory_modules"] = f"unavailable ({exc})"
    return info


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------
def _read(path):
    with open(path) as f:
        return f.read().strip()


def _parse_size(s):
    m = re.match(r"(\d+)\s*([KMG]?)", s)
    return int(m.group(1)) * {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[m.group(2)]


def probe_linux():
    info = {}
    txt = _read("/proc/cpuinfo")
    blocks = [b for b in txt.split("\n\n") if b.strip()]
    first = dict(re.findall(r"^([^\t:]+)\s*:\s*(.*)$", blocks[0], re.M))
    info["model_name"] = first.get("model name", platform.processor())
    fam, mod = int(first.get("cpu family", 0)), int(first.get("model", 0))
    info["cpuid_family"], info["cpuid_model"] = fam, mod
    try:
        info["base_frequency_mhz"] = int(_read("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency")) // 1000
    except OSError:
        m = re.search(r"@\s*([\d.]+)GHz", info["model_name"])
        info["base_frequency_mhz"] = int(float(m.group(1)) * 1000) if m else int(float(first.get("cpu MHz", 0)))
    core_ids = set()
    for b in blocks:
        d = dict(re.findall(r"^([^\t:]+)\s*:\s*(.*)$", b, re.M))
        core_ids.add((d.get("physical id", "0"), d.get("core id", d.get("processor"))))
    info["physical_cores"] = len(core_ids)
    info["logical_threads"] = os.cpu_count()
    info["cores_with_smt"] = info["physical_cores"] if info["logical_threads"] > info["physical_cores"] else 0
    caches, seen = [], set()
    base = "/sys/devices/system/cpu"
    for cpu in sorted(d for d in os.listdir(base) if re.fullmatch(r"cpu\d+", d)):
        cdir = f"{base}/{cpu}/cache"
        if not os.path.isdir(cdir):
            continue
        for idx in sorted(os.listdir(cdir)):
            if not idx.startswith("index"):
                continue
            p = f"{cdir}/{idx}"
            try:
                shared = _read(f"{p}/shared_cpu_list")
                lvl = int(_read(f"{p}/level"))
                typ = _read(f"{p}/type")
                key = (lvl, typ, shared)
                if key in seen:
                    continue
                seen.add(key)
                n_share = sum((int(b) - int(a) + 1) if "-" in r else 1
                              for r in shared.split(",") for a, _, b in [r.partition("-")] if r)
                caches.append({"level": lvl, "type": typ, "size": _parse_size(_read(f"{p}/size")),
                               "line": int(_read(f"{p}/coherency_line_size")), "sharing": n_share})
            except OSError:
                pass
    info["caches_raw"] = caches
    try:
        info["mem_total_kib"] = int(re.search(r"MemTotal:\s+(\d+)", _read("/proc/meminfo")).group(1))
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------
def probe_macos():
    def sc(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""
    info = {"model_name": sc("machdep.cpu.brand_string") or platform.processor(),
            "physical_cores": int(sc("hw.physicalcpu") or 0),
            "logical_threads": int(sc("hw.logicalcpu") or os.cpu_count())}
    f = sc("hw.cpufrequency")
    info["base_frequency_mhz"] = int(f) // 1_000_000 if f else None
    info["cores_with_smt"] = info["physical_cores"] if info["logical_threads"] > info["physical_cores"] else 0
    line = int(sc("hw.cachelinesize") or 64)
    caches = []
    for lvl, typ, k in ((1, "Data", "hw.l1dcachesize"), (1, "Instruction", "hw.l1icachesize"),
                        (2, "Unified", "hw.l2cachesize"), (3, "Unified", "hw.l3cachesize")):
        v = sc(k)
        if v and v != "0":
            caches.append({"level": lvl, "type": typ, "size": int(v), "line": line, "sharing": None})
    info["caches_raw"] = caches
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="8-digit student ID")
    args = ap.parse_args()
    if not re.fullmatch(r"\d{8}", args.student_id):
        print(f"WARNING: student id '{args.student_id}' is not 8 digits", file=sys.stderr)

    os.makedirs(RESULTS, exist_ok=True)
    system = platform.system()
    probe = {"Windows": probe_windows, "Linux": probe_linux, "Darwin": probe_macos}.get(system)
    raw = probe() if probe else {"model_name": platform.processor(), "logical_threads": os.cpu_count(), "caches_raw": []}

    # psutil cross-check when available (not required)
    try:
        import psutil
        raw["psutil_physical_cores"] = psutil.cpu_count(logical=False)
        raw["psutil_logical_threads"] = psutil.cpu_count(logical=True)
        raw["mem_total_bytes"] = psutil.virtual_memory().total
        fr = psutil.cpu_freq()
        if fr:
            raw["psutil_freq_mhz"] = {"current": fr.current, "min": fr.min, "max": fr.max}
    except Exception:
        pass

    phys, logi = raw.get("physical_cores"), raw.get("logical_threads")
    caches = summarize_caches(raw.get("caches_raw", []))
    l2_variants = caches.get("L2", {}).get("variants", [])
    profile = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "cpu": {
            "model_name": raw.get("model_name"),
            "microarchitecture": guess_uarch(raw.get("model_name"), raw.get("cpuid_family"), raw.get("cpuid_model")),
            "cpuid_family": raw.get("cpuid_family"), "cpuid_model": raw.get("cpuid_model"),
            "base_frequency_mhz": raw.get("base_frequency_mhz"),
            "physical_cores": phys,
            "logical_threads": logi,
            "smt_enabled": bool(phys and logi and logi > phys),
            "cores_with_smt": raw.get("cores_with_smt"),
            "hybrid_core_topology_suspected": len({v["size_bytes"] for v in caches.get("L1d", {}).get("variants", [])}) > 1
                                              or len({v["size_bytes"] for v in l2_variants}) > 1,
        },
        "cache": caches,
        "memory": {k: raw[k] for k in ("mem_total_bytes", "mem_total_kib", "memory_modules") if k in raw},
        "os": {"system": system, "release": platform.release(), "version": platform.version(),
               "kernel": platform.uname().release, "machine": platform.machine()},
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version(),
                   "compiler": platform.python_compiler(), "executable": sys.executable},
        "cross_checks": {k: raw[k] for k in ("psutil_physical_cores", "psutil_logical_threads", "psutil_freq_mhz") if k in raw},
        "raw_cache_instances": raw.get("caches_raw", []),
    }
    try:
        import numpy as np
        profile["python"]["numpy"] = np.__version__
    except ImportError:
        pass
    try:
        import numba
        profile["python"]["numba"] = numba.__version__
    except ImportError:
        profile["python"]["numba"] = None

    params = get_student_params(args.student_id)
    with open(os.path.join(RESULTS, "host_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    with open(os.path.join(RESULTS, "salted_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    c = profile["cpu"]
    print(f"CPU      : {c['model_name']}  [{c['microarchitecture']}]  base {c['base_frequency_mhz']} MHz")
    print(f"Cores    : {c['physical_cores']} physical / {c['logical_threads']} logical  (SMT={c['smt_enabled']}, hybrid={c['hybrid_core_topology_suspected']})")
    for lvl in ("L1d", "L2", "L3"):
        if lvl in caches:
            print(f"{lvl:<9}: " + ", ".join(f"{v['count']}x {v['size_human']}" for v in caches[lvl]["variants"]))
    print(f"Line     : {caches['cache_line_bytes']} B")
    print(f"OS/Python: {system} {platform.release()} / {platform.python_implementation()} {platform.python_version()}")
    print(f"Salted   : {params}")
    print("Wrote results/host_profile.json and results/salted_params.json")


if __name__ == "__main__":
    main()
