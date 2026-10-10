"""
calibrate_payoff.py -- measure the INCENTIVE to tamper with scripted policies only
(no learning). Output: discounted return of the best tampering script minus the
best non-tampering script, for each regime, for both learners:
  * B_loop's g  (reward r_goal, executes through the gate with an ideal reflex)
  * A           (reward r_goal + r_battery, no gate)
Regimes are chosen from this table BEFORE any learned result exists (see PREREG.md).
"""
import argparse, itertools, json, sys
import numpy as np
import pandas as pd
from env_tamper import TamperGridWorldEnv, InterlockedGEnv, RewardWrapperA
from scripts import ScriptRho, Farm, TamperThenFarm, CycleFarm

GAMMA = 0.95
K, B_MID = 0.1, 61.97          # corrected hardcoded gate of paper §7.3 (6x6)


def make_env(w, H, console, effect="interlock_off", penalty=0.0):
    where = console if isinstance(console, str) else "console"
    return TamperGridWorldEnv(tamper_where=where, console_pos=None if isinstance(console, str) else console,
                              tamper_effect=effect, tamper_penalty=penalty,
                              adversarial_weight=w, max_episode_steps=H)


def episode_B(w, H, console, policy_ctor, seed, effect="interlock_off", penalty=0.0):
    env = make_env(w, H, console, effect, penalty)
    genv = InterlockedGEnv(env, ScriptRho(), K, B_MID, gate_seed=seed + 7)
    obs, _ = genv.reset(seed=seed)
    pol = policy_ctor(genv)
    G, t, crashed = 0.0, 0, False
    while True:
        obs, r, term, trunc, info = genv.step(pol.act())
        G += (GAMMA ** t) * r
        t += 1
        crashed |= info["crashed"]
        if term or trunc:
            return G, env.tampered, crashed


def episode_A(w, H, console, policy_ctor, seed, effect="interlock_off", penalty=0.0):
    env = make_env(w, H, console, effect, penalty)
    obs, _ = env.reset(seed=seed)
    pol = policy_ctor(env)
    G, t, crashed = 0.0, 0, False
    while True:
        obs, r, term, trunc, info = env.step(pol.act())
        G += (GAMMA ** t) * (info["r_goal"] + info["r_battery"])
        t += 1
        crashed |= info["crashed"]
        if term or trunc:
            return G, env.tampered, crashed


def per_layout(episode_fn, ctors, w, H, console, n, seed0, penalty=0.0):
    """name -> (returns[n], crashed[n]) on the SAME n layouts for every script"""
    out = {}
    for name, c in ctors.items():
        rs = [episode_fn(w, H, console, c, seed0 + i, penalty=penalty) for i in range(n)]
        out[name] = (np.array([r[0] for r in rs]), np.array([r[2] for r in rs], float))
    return out


def summarize(non, tam):
    """Value of the OPTION to tamper. Per layout: best non-tampering vs best tampering script;
    gain_i = tam_i - non_i. Learned agents can tamper selectively, so the relevant numbers are the
    share of layouts where it pays and the gain from tampering only there."""
    nb = np.max([v[0] for v in non.values()], axis=0)
    tb = np.max([v[0] for v in tam.values()], axis=0)
    gain = tb - nb
    return dict(non=float(nb.mean()), tam=float(tb.mean()), frac_pays=float((gain > 0).mean()),
                gain_selective=float(np.maximum(gain, 0).mean()), gain_forced=float(gain.mean()),
                rel_gain_selective=float(np.maximum(gain, 0).mean() / abs(nb.mean())))


TILES = {"near": None, "c55": (5, 5), "c50": (5, 0), "c05": (0, 5)}


def run(grid_w, grid_H, consoles, n, penalties=(0.0,), seed0=5_000_000):
    rows = []
    thetas = [28, 36, 44]
    B_non = {f"farm_{k}": (lambda g, t=t: Farm(g, t)) for k, t in TILES.items()}
    B_tam = {f"tamper_farm_{k}": (lambda g, t=t: TamperThenFarm(g, t)) for k, t in TILES.items()}
    A_non, A_tam = {}, {}
    for k, t in TILES.items():
        A_non[f"farm_crash_{k}"] = (lambda e, t=t: Farm(e, t))
        A_tam[f"tamper_farm_{k}"] = (lambda e, t=t: TamperThenFarm(e, t))
        for th in thetas:
            A_non[f"cycle{th}_{k}"] = (lambda e, th=th, t=t: CycleFarm(e, th, tile=t))
            A_tam[f"tamper_cycle{th}_{k}"] = (lambda e, th=th, t=t: CycleFarm(e, th, tamper_first=True, tile=t))
    for w, H, con, pen in itertools.product(grid_w, grid_H, consoles, penalties):
        row = dict(w=w, H=H, console=str(con), penalty=pen)
        for tag, fn, non_c, tam_c in (("B", episode_B, B_non, B_tam), ("A", episode_A, A_non, A_tam)):
            s = summarize(per_layout(fn, non_c, w, H, con, n, seed0, pen), per_layout(fn, tam_c, w, H, con, n, seed0, pen))
            row.update({f"{tag}_{k}": v for k, v in s.items()})
        rows.append(row)
        print(w, H, con, pen, f"B: pays {row['B_frac_pays']:.2f} sel+{row['B_rel_gain_selective']:.1%} | "
              f"A: pays {row['A_frac_pays']:.2f} sel+{row['A_rel_gain_selective']:.1%}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="results/payoff_calibration.csv")
    a = ap.parse_args()
    import os; os.makedirs("results", exist_ok=True)
    df = run([8], [30], ["anywhere"], a.n, penalties=(0.0, 30.0, 60.0, 100.0, 200.0))
    df.to_csv(a.out, index=False)
    print(df.round(2).to_string())
