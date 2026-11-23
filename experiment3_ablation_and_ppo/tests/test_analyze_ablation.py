"""analyze_ablation.py on synthetic data with a KNOWN truth: it must reach the right verdict in each scenario."""
import io, contextlib
import numpy as np
import pandas as pd
import pytest

import analyze_ablation as AA

SEEDS = list(range(200, 220))
NETS, BUDS = ["[16]", "[2048]"], [10_000, 100_000, 500_000]
rng = np.random.default_rng(0)


def make(scenario):
    base = np.clip(rng.beta(0.8, 2.2, len(SEEDS)), 0, 0.85)          # per-seed gated baseline (reflex quality)
    rows = []
    for net in NETS:
        for bud in BUDS:
            f = {10_000: 0.0, 100_000: 0.7, 500_000: 1.0}[bud]        # how far tampering has been learned
            for i, s in enumerate(SEEDS):
                n = lambda sd=0.03: rng.normal(0, sd)
                common = dict(arch="B_loop", net=net, budget=bud, seed=s, mode="full", regimes_sha="x", sb3="", torch="",
                              gymnasium="", python="", crn_untampered_identical=True, placebo_rate=0.05)
                # ---- bonus regime: gate never bypassed
                tb = 0.95 if scenario != "no_manipulation" else 0.05
                crash_b = base[i] + (0.5 * f * (1 - base[i]) if scenario == "bonus_also_raises" else 0.0) + n()
                crash_b = float(np.clip(crash_b, 0, 1))
                rows.append(dict(common, regime="bonus", tamper_rate=tb if bud > 10_000 else 0.3, crash_rate=crash_b,
                                 crash_rate_crn=crash_b, crash_rate_cf=crash_b))
                # ---- pays regime
                bypass = f * 0.9 * (1 - base[i])                      # real bypass effect, bounded by the ceiling
                if scenario == "no_bypass_effect":
                    bypass = 0.0
                cf = float(np.clip(base[i] + n(), 0, 1))
                real = float(np.clip(cf + bypass + n(), 0, 1))
                if scenario == "bonus_also_raises":
                    cf = float(np.clip(crash_b + n(), 0, 1)); real = float(np.clip(cf + bypass * 0.5, 0, 1))
                rows.append(dict(common, regime="pays", tamper_rate=0.9 * f + 0.3 * (1 - f), crash_rate=real,
                                 crash_rate_crn=real, crash_rate_cf=cf))
    return pd.DataFrame(rows)


def run(scenario, tmp_path):
    p = tmp_path / f"{scenario}.csv"; make(scenario).to_csv(p, index=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        AA.main([str(p)], SEEDS)
    return buf.getvalue()


def test_real_bypass_effect_is_attributed(tmp_path):
    out = run("real", tmp_path)
    assert "BYPASS-ATTRIBUTED" in out

def test_no_bypass_effect_is_not_attributed(tmp_path):
    out = run("no_bypass_effect", tmp_path)
    assert "NOT ATTRIBUTED" in out and "E1" in out.split("unmet ->")[1].split("\n")[0]

def test_failed_manipulation_is_flagged_not_misread(tmp_path):
    out = run("no_manipulation", tmp_path)
    assert "MANIPULATION FAILED" in out

def test_crash_rise_in_bonus_regime_blocks_attribution(tmp_path):
    out = run("bonus_also_raises", tmp_path)
    assert "NOT ATTRIBUTED" in out

def test_integrity_assertions_fire(tmp_path):
    df = make("real")
    bad = df.copy(); bad.loc[bad.index[0], "crn_untampered_identical"] = False
    p = tmp_path / "bad1.csv"; bad.to_csv(p, index=False)
    with pytest.raises(AssertionError, match="INTEGRITY"):
        AA.load([str(p)], SEEDS)
    bad = df.copy(); i = bad[bad.regime == "bonus"].index[0]; bad.loc[i, "crash_rate_cf"] += 0.1
    p = tmp_path / "bad2.csv"; bad.to_csv(p, index=False)
    with pytest.raises(AssertionError, match="INTEGRITY"):
        AA.load([str(p)], SEEDS)
    bad = df[~((df.regime == "bonus") & (df.seed == 205) & (df.net == "[16]") & (df.budget == 10_000))]
    p = tmp_path / "bad3.csv"; bad.to_csv(p, index=False)
    with pytest.raises(AssertionError, match="seeds differ"):
        AA.load([str(p)], SEEDS)
