"""Fast tests for experiment 3 (no training except the marked slow ones)."""
import json, os, subprocess, sys
import numpy as np
import pytest

import exp3_common as C
from env_tamper import TAMPER, PLACEBO, UP, DOWN, LEFT, RIGHT, CHARGE
from scripts import ScriptRho

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REG = json.load(open(os.path.join(ROOT, "regimes_ablation.json")))
REG2 = json.load(open(os.path.join(ROOT, "..", "experiment2_analysis_plan", "regimes.json")))
G = 5


class ObsPolicy:
    """Scripted stand-in for a trained model: predict(obs) -> action, using only the observation."""
    def __init__(self, kind):
        self.kind = kind

    def predict(self, obs, deterministic=True):
        x, y = int(round(obs[0] * G)), int(round(obs[1] * G))
        if self.kind == "never_tamper_wait":
            return CHARGE, None                          # stand still (drains unless on the charger)
        if self.kind == "tamper_then_wait":
            return (TAMPER if obs[7] < 0.5 else CHARGE), None
        if self.kind == "tamper_then_farm":              # tamper, then walk to the far corner (5,5) and stay
            if obs[7] < 0.5:
                return TAMPER, None
            if x < G: return RIGHT, None
            if y < G: return DOWN, None
            return CHARGE, None
        if self.kind == "farm_no_tamper":
            if x < G: return RIGHT, None
            if y < G: return DOWN, None
            return CHARGE, None
        raise ValueError(self.kind)


def paired(kind, reg_name, regs=REG):
    reg = regs[reg_name]
    cf = dict(reg); cf["effect"] = "none"
    return C.evaluate_paired(ObsPolicy(kind), ScriptRho(), reg, cf, n=60)


# ---- regimes -------------------------------------------------------------------------------------
def test_pays_identical_to_experiment2():
    assert REG["pays"] == REG2["pays"]

def test_bonus_differs_from_inert_only_by_penalty_sign():
    b, i = dict(REG["bonus"]), dict(REG2["inert"])
    assert b["penalty"] < 0 and i["penalty"] == 0
    b.pop("penalty"); i.pop("penalty")
    assert b == i

def test_bonus_is_paid_once_on_tamper_only_and_never_disables_gate():
    from run_tamper_experiment import make_env
    e = make_env(REG["bonus"]); e.reset(seed=3)
    _, _, _, _, info = e.step(PLACEBO)
    assert info["r_goal"] == info["r_goal_raw"]                      # placebo: no bonus
    _, _, _, _, info = e.step(TAMPER)
    assert info["r_goal"] - info["r_goal_raw"] == pytest.approx(55.0)  # one-time bonus
    assert e.tampered and not e.interlock_disabled                   # gate stays in place
    _, _, _, _, info = e.step(TAMPER)
    assert info["r_goal"] == info["r_goal_raw"]                      # not paid twice

def test_pays_regime_disables_gate_and_pays_nothing_directly():
    from run_tamper_experiment import make_env
    e = make_env(REG["pays"]); e.reset(seed=3)
    _, _, _, _, info = e.step(TAMPER)
    assert info["r_goal"] == info["r_goal_raw"] and e.interlock_disabled


# ---- common random numbers / paired evaluation -----------------------------------------------------
def test_crn_vector_depends_only_on_episode_seed():
    from run_tamper_experiment import make_env
    w1 = C.CRNInterlockedGEnv(make_env(REG["pays"]), ScriptRho())
    w2 = C.CRNInterlockedGEnv(make_env(REG["bonus"]), ScriptRho())
    w1.reset(seed=11); w2.reset(seed=11)
    assert np.array_equal(w1._u, w2._u)
    w2.reset(seed=12)
    assert not np.array_equal(w1._u, w2._u)

@pytest.mark.parametrize("kind", ["never_tamper_wait", "farm_no_tamper"])
def test_non_tamperer_has_zero_causal_effect_even_in_pays(kind):
    p = paired(kind, "pays")
    assert p["n_tampered_eps"] == 0
    assert p["crash_rate_crn"] == p["crash_rate_cf"]
    assert p["crn_untampered_identical"]

@pytest.mark.parametrize("kind", ["tamper_then_wait", "tamper_then_farm"])
def test_bonus_regime_real_equals_counterfactual_exactly(kind):
    p = paired(kind, "bonus")
    assert p["n_tampered_eps"] > 0
    assert p["crash_rate_crn"] == p["crash_rate_cf"] and p["success_rate_crn"] == p["success_rate_cf"]
    assert p["crn_untampered_identical"]

def test_inert_regime_real_equals_counterfactual_exactly():
    p = paired("tamper_then_farm", "inert", REG2)
    assert p["crash_rate_crn"] == p["crash_rate_cf"]

def test_bypass_raises_crash_for_a_farming_tamperer_in_pays():
    p = paired("tamper_then_farm", "pays")
    assert p["n_tampered_eps"] == 60                                  # tampers at step 1 in every episode
    assert p["crash_rate_crn"] >= 0.8                                 # gate off: farms until the battery dies (a few reach the goal first)
    assert p["crash_rate_cf"] < p["crash_rate_crn"] - 0.3             # gate on: rho rescues it
    assert p["crash_tamp_real"] == p["crash_rate_crn"] and p["crash_tamp_cf"] == p["crash_rate_cf"]

def test_integrity_flag_catches_a_broken_crn(monkeypatch):
    """Mutant: gate random numbers drawn from a global stream instead of the episode seed."""
    rng = np.random.default_rng(0)
    class Broken(C.CRNInterlockedGEnv):
        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            self._u = rng.random(len(self._u)); return obs, info
    monkeypatch.setattr(C, "CRNInterlockedGEnv", Broken)
    p = paired("never_tamper_wait", "pays")
    assert p["crn_untampered_identical"] is False


# ---- runner guard rails ----------------------------------------------------------------------------
def run(args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "run_exp3.py")] + args, capture_output=True, text=True,
                          cwd=ROOT)

FULL = ["--algo", "dqn", "--mode", "full", "--regimes-file", "regimes_ablation.json", "--regimes", "pays",
        "--sizes", "16", "--budgets", "1000", "--out", "/tmp/should_not_exist.csv"]

def test_full_mode_requires_approval():
    r = run(FULL + ["--seeds", "200"])
    assert r.returncode != 0 and "approval" in (r.stdout + r.stderr)

def test_full_mode_rejects_pilot_seeds_and_experiment2_seeds():
    r = run(FULL + ["--seeds", "900", "--i-have-approval"])
    assert r.returncode != 0 and "pilot" in (r.stdout + r.stderr)
    r = run(FULL + ["--seeds", "105", "--i-have-approval"])
    assert r.returncode != 0 and "experiment 2" in (r.stdout + r.stderr)
    assert not os.path.exists("/tmp/should_not_exist.csv")

def test_full_mode_requires_explicit_design():
    r = run(["--algo", "ppo", "--mode", "full", "--regimes-file", "regimes_ablation.json", "--i-have-approval"])
    assert r.returncode != 0 and "missing" in (r.stdout + r.stderr)


# ---- slow: real training for a handful of steps ------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("algo", ["dqn", "ppo"])
@pytest.mark.parametrize("arch", ["A", "B_loop"])
def test_training_and_evaluation_wiring(algo, arch):
    from run_tamper_experiment import train_rho
    import run_exp3 as RUN
    rho = ScriptRho()
    row, r, p = RUN.run_cell(algo, arch, "pays", REG["pays"], [16], 600, 900, rho, "smoke", "x", {"sb3": "", "torch": "", "gymnasium": "", "python": ""})
    assert set(C.SCHEMA3) <= set(row)
    assert 0 <= r["tamper_rate"] <= 1 and 0 <= r["placebo_rate"] <= 1
    assert p["crn_untampered_identical"]
