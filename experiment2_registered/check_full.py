"""
check_full.py -- INTEGRITY of finished full-run CSVs. It looks at no outcome and runs no test.
    python check_full.py results/tamper_full_*.csv              # expects seeds 100-109 (PREREG_v2)
    python check_full.py results/tamper_full_*.csv --seeds 0-9  # check a partial set
Exit code 0 only if the design for the requested seeds is complete and consistent.
"""
import argparse, itertools, sys
import pandas as pd

ARCHS = ["A", "B_loop"]; REGIMES = ["pays", "costly", "inert"]
NETS = ["[16]", "[256]", "[2048]"]; BUDGETS = [10_000, 100_000, 500_000]


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--seeds", default="100-109")
    a = ap.parse_args(); want = parse_seeds(a.seeds)
    d = pd.concat([pd.read_csv(f).assign(_file=f) for f in a.files], ignore_index=True)
    ok = True
    def chk(c, msg):
        nonlocal ok; print(("PASS  " if c else "FAIL  ") + msg); ok &= bool(c)
    key = ["arch", "regime", "net", "budget", "seed"]
    print(f"rows={len(d)}  files={len(a.files)}  seeds present={sorted(int(x) for x in d.seed.unique())}")
    chk(set(d["mode"]) == {"full"}, f"mode == full only (found {set(d['mode'])})")
    chk(d.regimes_sha.nunique() == 1, f"one regimes.json hash ({set(d.regimes_sha)})")
    v = d[["sb3", "torch", "gymnasium", "python"]].drop_duplicates()
    chk(len(v) == 1, f"one library-version set: {v.values.tolist()}")
    dup = d.duplicated(key, keep=False)
    chk(not dup.any(), f"no duplicate runs ({int(dup.sum())} rows involved)")
    chk(not d.isna().any().any(), "no NaN")
    rc = ["tamper_rate", "placebo_rate", "tamper_early_rate", "placebo_early_rate", "crash_rate", "success_rate", "mean_gate_rho_frac"]
    chk(((d[rc] >= 0) & (d[rc] <= 1)).all().all(), "rates within [0,1]")
    got = set(map(tuple, d[key].astype({"seed": int, "budget": int}).values.tolist()))
    exp = set(itertools.product(ARCHS, REGIMES, NETS, BUDGETS, want))
    miss, extra = sorted(exp - got), sorted(got - exp)
    chk(not miss, f"design complete for seeds {a.seeds}: missing {len(miss)} of {len(exp)} runs" + (f"; first: {miss[:3]}" if miss else ""))
    chk(not extra, f"no runs outside the requested design: extra {len(extra)}" + (f"; first: {extra[:3]}" if extra else ""))
    chk((d[d.arch == "A"].mean_gate_rho_frac == 0).all(), "A never uses the gate")
    print(f"\ncompute time in the CSVs: {d.train_s.sum()/3600:.1f} core-hours; "
          f"longest single run {d.train_s.max()/60:.1f} min")
    print("\nINTEGRITY OK" if ok else "\nINTEGRITY PROBLEMS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
