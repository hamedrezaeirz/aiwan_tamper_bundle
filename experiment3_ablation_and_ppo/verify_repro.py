"""
verify_repro.py -- does re-training with the frozen experiment-2 code reproduce the PUBLISHED rows exactly?
Meant to be run ON THE MACHINE THAT WILL RUN EXPERIMENT 3 (pinned requirements_vps.txt). It retrains a few small
cells of experiment 2 (seeds 100-119, which are NOT used by experiment 3) and compares every evaluation number
with the published CSV row. Nothing here enters any registered result.

    python verify_repro.py                      # default: B_loop, pays and inert, [16], 10k steps, seed 100 (~2 min)
    python verify_repro.py --seeds 100 101 --sizes 16 256 --budgets 10000 100000 --archs A B_loop

Why it matters: experiment 3's within-agent comparisons do NOT depend on reproducing old agents, but if this check
passes, the paper can additionally say that the retrained 'pays' agents of the ablation are the same procedure as
experiment 2's. If it fails, the paper says so: the cause is the library/CPU difference (in the development sandbox, with
SB3 2.9.2 / torch 2.15 / Python 3.12 instead of 2.9.0 / 2.14.1 / 3.10, even one 10k-step cell did NOT reproduce).
"""
import argparse, glob, os, sys, time
import pandas as pd
import exp3_common as C
R2 = C.R2

FIELDS = ["tamper_rate", "placebo_rate", "tamper_early_rate", "placebo_early_rate", "crash_rate", "success_rate",
          "mean_farm_steps", "mean_len", "mean_return_goal", "mean_gate_rho_frac"]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=[100])
ap.add_argument("--sizes", nargs="+", type=int, default=[16])
ap.add_argument("--budgets", nargs="+", type=int, default=[10_000])
ap.add_argument("--archs", nargs="+", default=["B_loop"])
ap.add_argument("--regimes", nargs="+", default=["pays", "inert"])
a = ap.parse_args()

pub = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(C.EXP2, "results", "tamper2_full_*.csv"))])
regimes, _ = R2.load_regimes()
n_ok = n_all = 0
for seed in a.seeds:
    rho = R2.train_rho(seed)
    for reg_name in a.regimes:
        for s in a.sizes:
            for bud in a.budgets:
                for arch in a.archs:
                    t0 = time.time()
                    m = C.train_agent("dqn", arch, regimes[reg_name], [s], bud, seed, rho)
                    r = R2.evaluate(arch, regimes[reg_name], m, rho, seed)
                    ref = pub[(pub.arch == arch) & (pub.regime == reg_name) & (pub.net == str([s])) &
                              (pub.budget == bud) & (pub.seed == seed)].iloc[0]
                    diffs = {k: (r[k], ref[k]) for k in FIELDS if abs(r[k] - ref[k]) > 1e-9}
                    n_all += 1; n_ok += not diffs
                    print(f"{arch:6s} {reg_name:6s} [{s}] {bud:>7} s{seed}: " +
                          ("IDENTICAL to the published row" if not diffs else
                           "DIFFERS: " + ", ".join(f"{k} new={v[0]:.4f} published={v[1]:.4f}" for k, v in list(diffs.items())[:4])) +
                          f"  ({time.time()-t0:.0f}s)", flush=True)
print(f"\n{n_ok}/{n_all} cells reproduce the published row exactly. versions: {R2.versions()}")
sys.exit(0 if n_ok == n_all else 1)
