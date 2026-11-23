"""
check_exp3.py -- INTEGRITY of finished experiment-3 CSVs. Looks at no outcome and runs no test.
The design is stated explicitly on the command line (it is also written in PREREG_exp3.md):

  python check_exp3.py results/ablation_dqn_*.csv --algo dqn --archs B_loop --regimes pays bonus \
         --sizes 16 2048 --budgets 10000 100000 500000 --seeds 200-219
  python check_exp3.py results/ppo_*.csv --algo ppo --archs A B_loop --regimes pays costly inert \
         --sizes 16 2048 --budgets 10000 100000 500000 --seeds 300-319
Exit code 0 only if the design is complete and consistent.
"""
import argparse, itertools, sys
import pandas as pd


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--algo", required=True, choices=["dqn", "ppo"])
    ap.add_argument("--archs", nargs="+", required=True)
    ap.add_argument("--regimes", nargs="+", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, required=True)
    ap.add_argument("--budgets", nargs="+", type=int, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--mode", default="full")
    a = ap.parse_args(argv)
    want = parse_seeds(a.seeds)
    d = pd.concat([pd.read_csv(f) for f in a.files], ignore_index=True)
    ok = True

    def chk(c, msg):
        nonlocal ok; print(("PASS  " if c else "FAIL  ") + msg); ok &= bool(c)

    key = ["arch", "regime", "net", "budget", "seed"]
    print(f"rows={len(d)} files={len(a.files)} seeds present={sorted(int(x) for x in d.seed.unique())}")
    chk(set(d["mode"]) == {a.mode}, f"mode == {a.mode} only (found {set(d['mode'])})")
    chk(set(d["algo"]) == {a.algo}, f"algo == {a.algo} only (found {set(d['algo'])})")
    chk(d.regimes_sha.nunique() == 1, f"one regimes file hash ({set(d.regimes_sha)})")
    v = d[["sb3", "torch", "gymnasium", "python"]].drop_duplicates()
    chk(len(v) == 1, f"one library-version set: {v.values.tolist()}")
    dup = d.duplicated(key, keep=False)
    chk(not dup.any(), f"no duplicate runs ({int(dup.sum())} rows involved)")
    crn = ["crash_rate_crn", "crash_rate_cf", "success_rate_crn", "success_rate_cf", "crash_tamp_real", "crash_tamp_cf"]
    chk(not d.drop(columns=crn).drop(columns=["model_path"]).isna().any().any(), "no NaN outside the A-only CRN columns")
    chk(not d[d.arch == "B_loop"][crn].isna().any().any(), "B_loop rows have the paired-evaluation columns")
    rc = ["tamper_rate", "placebo_rate", "tamper_early_rate", "placebo_early_rate", "crash_rate", "success_rate", "mean_gate_rho_frac"]
    chk(((d[rc] >= 0) & (d[rc] <= 1)).all().all(), "rates within [0,1]")
    got = set(map(tuple, d[key].astype({"seed": int, "budget": int}).values.tolist()))
    exp = set(itertools.product(a.archs, a.regimes, [f"[{s}]" for s in a.sizes], a.budgets, want))
    miss, extra = sorted(exp - got), sorted(got - exp)
    chk(not miss, f"design complete: missing {len(miss)} of {len(exp)} runs" + (f"; first: {miss[:3]}" if miss else ""))
    chk(not extra, f"no runs outside the stated design: extra {len(extra)}" + (f"; first: {extra[:3]}" if extra else ""))
    if a.mode == "full":
        chk(not (set(d.seed) & set(range(100, 120))), "no seed from experiment 2 (100-119)")
        chk(not (set(d.seed) & {900, 901}), "no pilot/smoke seed (900, 901)")
    chk((d[d.arch == "A"].mean_gate_rho_frac == 0).all(), "A never uses the gate")
    b = d[d.arch == "B_loop"]
    chk(b.crn_untampered_identical.astype(bool).all(), "B_loop: untampered episodes identical in real vs counterfactual replay")
    nb = b[b.effect == "none"]
    chk((nb.crash_rate_crn == nb.crash_rate_cf).all(), "B_loop, regimes where TAMPER is harmless: real == counterfactual exactly")
    print(f"\ncompute time in the CSVs: {d.train_s.sum()/3600:.1f} core-hours; longest single run {d.train_s.max()/60:.1f} min")
    print("\nINTEGRITY OK" if ok else "\nINTEGRITY PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
