"""
run_exp3.py
-----------
Runner for experiment 3 (see PREREG_exp3.md).  One script, two uses:

  ABLATION (DQN)  --algo dqn --regimes-file regimes_ablation.json --archs B_loop --regimes pays bonus ...
  REPLICATION     --algo ppo --regimes-file ../experiment2_analysis_plan/regimes.json ...

Every trained agent is evaluated (i) exactly as in experiment 2 (R2.evaluate, 200 fixed layouts) and, for
B_loop, (ii) in a PAIRED way (common random numbers): real environment vs counterfactual environment in which
TAMPER is harmless. Both sets of numbers go into the same CSV row.

  python run_exp3.py --algo dqn --mode smoke --regimes-file regimes_ablation.json     # seconds, wiring only
  python run_exp3.py --algo dqn --mode pilot --regimes-file regimes_ablation.json     # minutes, pilot seeds 900/901
  python run_exp3.py --algo dqn --mode full  --i-have-approval --regimes-file ... --archs ... --regimes ... \
                     --sizes ... --budgets ... --seeds ... --out results/xxx.csv [--save-models results/models]
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp3_common as C                                   # noqa: E402
R2 = C.R2

PILOT_SEEDS, SMOKE_SEEDS = [900, 901], [900]              # never used in a registered run
DEFAULTS = {"smoke": dict(sizes=[16], budgets=[1_000]), "pilot": dict(sizes=[256], budgets=[100_000])}


def cf_regime(reg):
    """Counterfactual world: same regime, but TAMPER never disables anything (the gate stays in place)."""
    r = dict(reg); r["effect"] = "none"; return r


def run_cell(algo, arch, reg_name, reg, net, budget, seed, rho, mode, sha, ver, save_dir=None):
    t0 = time.time()
    m = C.train_agent(algo, arch, reg, net, budget, seed, rho)
    train_s = round(time.time() - t0, 1)
    r = R2.evaluate(arch, reg, m, rho, seed)               # as in experiment 2
    if arch == "B_loop":
        p = C.evaluate_paired(m, rho, reg, cf_regime(reg))
    else:                                                  # A: TAMPER changes only the reward, never the physics
        p = dict(crash_rate_crn=np.nan, crash_rate_cf=np.nan, success_rate_crn=np.nan, success_rate_cf=np.nan,
                 n_tampered_eps=-1, crash_tamp_real=np.nan, crash_tamp_cf=np.nan, crn_untampered_identical=True)
    path = ""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{algo}_{arch}_{reg_name}_{net[0]}_{budget}_s{seed}.zip")
        m.save(path)
    row = dict(arch=arch, regime=reg_name, net=str(net), budget=budget, seed=seed, **r,
               w=reg["w"], H=reg["H"], where=reg.get("where", "anywhere"), effect=reg["effect"],
               train_s=train_s, mode=mode, regimes_sha=sha, **ver, algo=algo, model_path=path, **p)
    return row, r, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    ap.add_argument("--mode", choices=["smoke", "pilot", "full"], required=True)
    ap.add_argument("--regimes-file", required=True)
    ap.add_argument("--regimes", nargs="*", default=None)
    ap.add_argument("--archs", nargs="*", default=["A", "B_loop"])
    ap.add_argument("--sizes", nargs="*", type=int, default=None)
    ap.add_argument("--budgets", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-models", default=None)
    ap.add_argument("--i-have-approval", action="store_true")
    a = ap.parse_args()

    if a.mode == "full":
        if not a.i_have_approval:
            sys.exit("Refusing --mode full without --i-have-approval (after PREREG_exp3.md is approved and frozen).")
        missing = [n for n in ("regimes", "sizes", "budgets", "seeds", "out") if getattr(a, n) in (None, [])]
        if missing:
            sys.exit(f"--mode full needs the design stated explicitly; missing: {missing}")
        bad = set(a.seeds) & (set(PILOT_SEEDS) | set(SMOKE_SEEDS))
        if bad:
            sys.exit(f"seeds {sorted(bad)} are pilot/smoke seeds and may not enter a registered run")
        low = [s for s in a.seeds if 100 <= s <= 119]
        if low:
            sys.exit(f"seeds {low} were used in experiment 2; experiment 3 needs new seeds")
    sizes = [[s] for s in (a.sizes or DEFAULTS.get(a.mode, {}).get("sizes", []))]
    budgets = a.budgets or DEFAULTS.get(a.mode, {}).get("budgets", [])
    seeds = a.seeds or {"smoke": SMOKE_SEEDS, "pilot": PILOT_SEEDS}.get(a.mode)
    out = a.out or os.path.join(HERE, "results", f"exp3_{a.algo}_{a.mode}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    regimes, sha = C.load_regimes(a.regimes_file)
    names = a.regimes or list(regimes.keys())
    ver = R2.versions()
    print(f"algo={a.algo} mode={a.mode} regimes={names} (sha {sha}) archs={a.archs} sizes={sizes} budgets={budgets} "
          f"seeds={seeds}", flush=True)
    if a.algo == "ppo":
        print("PPO_HP =", C.PPO_HP, flush=True)

    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out)
        done = set(zip(d.arch, d.regime, d.net.astype(str), d.budget, d.seed))
    t_all = time.time()
    for seed in seeds:
        rho = R2.train_rho(seed)                                # one reflex per seed (DQN, as in experiment 2)
        if a.save_models:
            os.makedirs(a.save_models, exist_ok=True); rho.save(os.path.join(a.save_models, f"rho_s{seed}.zip"))
        for reg_name in names:
            reg = regimes[reg_name]
            for net in sizes:
                for budget in budgets:
                    for arch in a.archs:
                        if (arch, reg_name, str(net), budget, seed) in done:
                            continue
                        row, r, p = run_cell(a.algo, arch, reg_name, reg, net, budget, seed, rho, a.mode, sha, ver,
                                             a.save_models)
                        pd.DataFrame([row])[C.SCHEMA3].to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                        extra = (f" crash_real={p['crash_rate_crn']:.2f} crash_cf={p['crash_rate_cf']:.2f}"
                                 if arch == "B_loop" else "")
                        print(f"[{time.time()-t_all:7.0f}s] {a.algo} {arch:6s} {reg_name:6s} {net} {budget:>7} s{seed} "
                              f"tamper={r['tamper_rate']:.2f} placebo={r['placebo_rate']:.2f} crash={r['crash_rate']:.2f}"
                              f"{extra} ({row['train_s']}s)", flush=True)


if __name__ == "__main__":
    main()
