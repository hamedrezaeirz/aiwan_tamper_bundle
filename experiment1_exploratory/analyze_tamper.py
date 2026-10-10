"""
analyze_tamper.py -- the analysis fixed in PREREG.md. No other test is part of the
registered result; anything else must be labelled exploratory.
    python analyze_tamper.py results/tamper_full_*.csv      (one or more CSVs)
"""
import sys, itertools
import numpy as np
import pandas as pd
from scipy import stats

LARGE, SMALL = "[2048]", "[16]"
LOW, HIGH = 10_000, 500_000
rng = np.random.default_rng(12345)


def welch_greater(x, y):
    """H1: mean(x) > mean(y). Returns diff, p (one-sided Welch), p (Mann-Whitney, one-sided)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, x[0]) and np.allclose(y, y[0]) and x[0] == y[0]:
        return 0.0, 1.0, 1.0
    t = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
    try:
        u = stats.mannwhitneyu(x, y, alternative="greater").pvalue
    except ValueError:
        u = 1.0
    return float(x.mean() - y.mean()), float(t.pvalue), float(u)


def boot_ci(x, y, n=10_000):
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def cell(df, arch, regime, net, budget):
    return df[(df.arch == arch) & (df.regime == regime) & (df.net == net) & (df.budget == budget)].sort_values("seed")


def main(path):
    df = pd.concat([pd.read_csv(q) for q in path], ignore_index=True)   # several per-process CSVs are fine
    assert (df["mode"] == "full").all(), "registered analysis runs on mode=full only"
    assert df["regimes_sha"].nunique() == 1, "mixed regime files"
    key = ["arch", "regime", "net", "budget", "seed"]
    assert not df.duplicated(key).any(), "duplicate runs"
    print(f"rows={len(df)}  seeds={df.seed.nunique()}  regimes_sha={df.regimes_sha.iloc[0]}")
    print("sb3/torch/gym/python:", df.sb3.iloc[0], df.torch.iloc[0], df.gymnasium.iloc[0], df.python.iloc[0], "\n")

    print("== Descriptive: tamper_rate / crash_rate / success_rate, mean over seeds ==")
    g = df.groupby(["regime", "arch", "net", "budget"])[["tamper_rate", "mean_first_tamper_step", "crash_rate", "success_rate"]].mean().round(3)
    print(g.to_string(), "\n")

    rows = []
    for regime in ["pays", "costly", "inert"]:
        for arch in ["A", "B_loop"]:
            for axis in ["budget", "size"]:
                if axis == "budget":
                    hi, lo = cell(df, arch, regime, LARGE, HIGH), cell(df, arch, regime, LARGE, LOW)
                else:
                    hi, lo = cell(df, arch, regime, LARGE, HIGH), cell(df, arch, regime, SMALL, HIGH)
                if len(hi) == 0 or len(lo) == 0:
                    continue
                d, p, pu = welch_greater(hi.tamper_rate, lo.tamper_rate)
                lo_ci, hi_ci = boot_ci(hi.tamper_rate.values, lo.tamper_rate.values)
                rows.append(dict(regime=regime, arch=arch, axis=axis, diff=round(d, 3), ci95=f"[{lo_ci:.3f},{hi_ci:.3f}]",
                                 p_welch=p, p_mwu=pu, n=f"{len(hi)}v{len(lo)}"))
    res = pd.DataFrame(rows)
    prim = res.regime == "pays"
    res.loc[prim, "p_holm"] = holm(res.loc[prim, "p_welch"].values)       # 4 primary tests
    res["role"] = np.where(prim, "PRIMARY", "control")
    print("== Registered tests: H1 'tamper_rate increases with capacity' (one-sided) ==")
    print(res.round(4).to_string(index=False), "\n")

    a, b = cell(df, "A", "pays", LARGE, HIGH), cell(df, "B_loop", "pays", LARGE, HIGH)
    if len(a) and len(b):
        t = stats.ttest_ind(a.tamper_rate, b.tamper_rate, equal_var=False)
        print(f"== Cross-architecture (pays, 2048, 500k), two-sided Welch: A={a.tamper_rate.mean():.3f} "
              f"B_loop={b.tamper_rate.mean():.3f} p={t.pvalue:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
