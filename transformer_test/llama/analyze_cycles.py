#!/usr/bin/env python3
# Parse llama3_1B_{1..11}[.log|_notcm.log|_nofuse.log] and produce:
#   1) cumulative cycles per test
#   2) first-difference (per-operator) cycles
#   3) TCM benefit and fusion benefit per operator
#
# Usage: python3 analyze_cycles.py <log_dir>
# Expects files named:  llama3_1B_1.log, llama3_1B_1_notcm.log, llama3_1B_1_nofuse.log, ...
# Each log should contain a line matching:  total cycles:<int>

import sys, os, re

OP_NAMES = [
    "1: pre_RMSNorm+SmoothQuant",   # base cumulative
    "2: +proj_q",
    "3: +proj_k",
    "4: +proj_v",
    "5: +Q.K^T+softmax",
    "6: +scores.V",
    "7: +smoothquant(attn)+proj_o",
    "8: +ffn_RMSNorm+smoothquant",
    "9: +ffn_gate+SILU",
    "10: +ffn_up+HADAMARD",
    "11: +ffn_down+RESADD",
]

CYCLE_RE = re.compile(r"total\s+cycles\s*[:=]\s*(\d+)", re.IGNORECASE)

def parse_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            m = CYCLE_RE.search(line)
            if m:
                return int(m.group(1))
    return None

def collect(log_dir, suffix):
    out = []
    for i in range(1, 12):
        name = f"llama3_1B_{i}{suffix}.log"
        cycles = parse_log(os.path.join(log_dir, name))
        out.append(cycles)
    return out

def fmt(x):
    if x is None:  return "    N/A"
    return f"{x:>8}"

def diff(cum):
    d = [cum[0]]
    for i in range(1, len(cum)):
        if cum[i] is None or cum[i-1] is None:
            d.append(None)
        else:
            d.append(cum[i] - cum[i-1])
    return d

def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    base   = collect(log_dir, "")
    notcm  = collect(log_dir, "_notcm")
    nofuse = collect(log_dir, "_nofuse")

    print(f"\n=== Cumulative cycles (from {log_dir}) ===")
    print(f"{'Test':40}  {'base':>8} {'notcm':>8} {'nofuse':>8}")
    for name, b, n, f in zip(OP_NAMES, base, notcm, nofuse):
        print(f"{name:40}  {fmt(b)} {fmt(n)} {fmt(f)}")

    b_d, n_d, f_d = diff(base), diff(notcm), diff(nofuse)

    print(f"\n=== Per-operator cycles (first difference) ===")
    print(f"{'Op':40}  {'base':>8} {'notcm':>8} {'nofuse':>8}  {'TCM_gain':>9} {'fuse_gain':>9}")
    for name, b, n, f in zip(OP_NAMES, b_d, n_d, f_d):
        tcm_gain = f"{n/b:>8.2f}x" if b and n else "     N/A"
        fuse_gain = f"{f/b:>8.2f}x" if b and f else "     N/A"
        print(f"{name:40}  {fmt(b)} {fmt(n)} {fmt(f)}  {tcm_gain} {fuse_gain}")

    # Total block accounting
    if all(base) and all(notcm) and all(nofuse):
        tb, tn, tf = base[-1], notcm[-1], nofuse[-1]
        print(f"\n=== Total llama block ===")
        print(f"  base   : {tb:>10}  cycles")
        print(f"  notcm  : {tn:>10}  cycles   ({tn/tb:.2f}x, TCM saves {(tn-tb)/tn*100:.1f}%)")
        print(f"  nofuse : {tf:>10}  cycles   ({tf/tb:.2f}x, fusion saves {(tf-tb)/tf*100:.1f}%)")

if __name__ == "__main__":
    main()
