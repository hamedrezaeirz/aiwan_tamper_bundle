"""
calibrate_bonus.py -- choose the size of the one-time TAMPER bonus of the 'bonus' regime WITHOUT any learning.

Rule (fixed before any learned result): the bonus b is the value for which the scripted gain from tampering in
the bonus regime (gate stays in place, TAMPER pays +b once) equals the scripted gain from tampering in 'pays'
(gate bypassed, no bonus), measured exactly as in experiment 1's calibrate_payoff.py (B_loop's g behind the
corrected gate, ideal reflex, 150 layouts, gamma 0.95, 'selective' gain = mean over layouts of max(0, best
tampering script - best non-tampering script)).

Step 1 reproduces the published number for 'pays' (experiment 1 bundle: B_gain_selective = 48.839) as a check of
this re-implementation; step 2 scans b.
    python calibrate_bonus.py
"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import exp3_common as C                      # noqa: F401  (sets sys.path to the frozen experiment-2 code)
from env_tamper import TamperGridWorldEnv, InterlockedGEnv
from scripts import ScriptRho, Farm, TamperThenFarm

GAMMA, K, B_MID, H, W = 0.95, 0.1, 61.97, 30, 8.0
N, SEED0 = 150, 5_000_000
TILES = {"near": None, "c55": (5, 5), "c50": (5, 0), "c05": (0, 5)}


def episode(policy_ctor, seed, effect, penalty):
    env = TamperGridWorldEnv(tamper_where="anywhere", console_pos=None, tamper_effect=effect, tamper_penalty=penalty,
                             adversarial_weight=W, max_episode_steps=H)
    g = InterlockedGEnv(env, ScriptRho(), K, B_MID, gate_seed=seed + 7)
    g.reset(seed=seed)
    pol = policy_ctor(g)
    G, t = 0.0, 0
    while True:
        _, r, term, trunc, info = g.step(pol.act())
        G += GAMMA ** t * r; t += 1
        if term or trunc:
            return G


def best(ctors, effect, penalty):
    R = np.array([[episode(c, SEED0 + i, effect, penalty) for i in range(N)] for c in ctors.values()])
    return R.max(axis=0)


NON = {k: (lambda g, t=t: Farm(g, t)) for k, t in TILES.items()}
TAM = {k: (lambda g, t=t: TamperThenFarm(g, t)) for k, t in TILES.items()}


def gain(effect, penalty):
    non = best(NON, effect, penalty)
    tam = best(TAM, effect, penalty)
    g = tam - non
    return float(non.mean()), float(tam.mean()), float(np.maximum(g, 0).mean())


if __name__ == "__main__":
    n0, t0, g0 = gain("interlock_off", 0.0)
    print(f"pays : non {n0:.3f}  tam {t0:.3f}  gain_selective {g0:.3f}   (published experiment 1: 46.946 / 95.785 / 48.839)")
    print("bonus: b      non       tam    gain_selective   |gain - pays gain|")
    rows = []
    for b in (30, 40, 45, 48, 49, 50, 52, 55, 60, 80):
        n1, t1, g1 = gain("none", -float(b))
        rows.append((abs(g1 - g0), b)); print(f"     {b:4d}  {n1:8.3f} {t1:9.3f} {g1:12.3f}   {abs(g1-g0):8.3f}")
    print("closest integer bonus:", min(rows)[1])
