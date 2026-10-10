"""
plot_tamper2.py -- DESCRIPTIVE figures of experiment 2. The list of figures is fixed here, before the data are complete;
no test is run and nothing here is a registered result (the only registered analysis is analyze_tamper2.py).

    python plot_tamper2.py results/tamper2_full_*.csv --out figures

It REFUSES to run unless the registered design is complete (1080 runs, seeds 100-119), so nobody looks at partial results.
--allow-partial exists only for testing the script on synthetic data.
"""
import argparse, itertools, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARCHS = ["A", "B_loop"]; REGIMES = ["pays", "costly", "inert"]
NETS = ["[16]", "[256]", "[2048]"]; BUDGETS = [10_000, 100_000, 500_000]; BLAB = ["10k", "100k", "500k"]
SEEDS = list(range(100, 120))
COL = {"[16]": "#4C78A8", "[256]": "#F58518", "[2048]": "#54A24B"}
rng = np.random.default_rng(2026)


def ci(x, n=5000):
    x = np.asarray(x, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return float(x.mean()), float(x.mean()), float(x.mean())
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def cell(df, arch, regime, net, budget, col):
    return df[(df.arch == arch) & (df.regime == regime) & (df.net == net) & (df.budget == budget)].sort_values("seed")[col].values


def grid(title, ylabel, ylim, ref_lines=None):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=12)
    for i, a in enumerate(ARCHS):
        for j, r in enumerate(REGIMES):
            ax = axs[i, j]; ax.set_title(f"{a}  -  {r}", fontsize=10)
            ax.set_xticks(range(3)); ax.set_xticklabels(BLAB); ax.set_ylim(*ylim); ax.grid(alpha=.25)
            if j == 0: ax.set_ylabel(ylabel)
            if i == 1: ax.set_xlabel("training budget (steps)")
            for y, ls, c in (ref_lines or []):
                ax.axhline(y, ls=ls, color=c, lw=1)
    return fig, axs


def line_panel(ax, df, arch, regime, col, nets=NETS, **kw):
    for k, net in enumerate(nets):
        m, lo, hi = zip(*[ci(cell(df, arch, regime, net, b, col)) for b in BUDGETS])
        x = np.arange(3) + (k - 1) * 0.06
        ax.errorbar(x, m, yerr=[np.array(m) - lo, np.array(hi) - m], color=COL[net], marker="o", ms=4, lw=1.4, capsize=2, label=net, **kw)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", default="figures")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(f) for f in a.files], ignore_index=True)
    key = ["arch", "regime", "net", "budget", "seed"]
    want = set(itertools.product(ARCHS, REGIMES, NETS, BUDGETS, SEEDS))
    got = set(map(tuple, df[key].astype({"seed": int, "budget": int}).values.tolist()))
    complete = (got == want) and not df.duplicated(key).any() and (df["mode"] == "full").all()
    if not complete and not a.allow_partial:
        sys.exit(f"REFUSED: the registered design is not complete ({len(got & want)} of {len(want)} runs). "
                 "Wait for DONE.txt = COMPLETE; this guard keeps partial results unseen.")
    os.makedirs(a.out, exist_ok=True)
    df["D"] = df.tamper_rate - df.placebo_rate
    rows = []

    # Fig 1: D by budget, one line per network size
    fig, axs = grid("Experiment 2: D = tamper_rate - placebo_rate (mean over seeds, 95% bootstrap CI)", "D", (-1.05, 1.05),
                    [(0, "-", "grey")])
    for i, ar in enumerate(ARCHS):
        for j, rg in enumerate(REGIMES):
            line_panel(axs[i, j], df, ar, rg, "D")
            if rg == "pays": axs[i, j].axhline(0.20, ls="--", color="k", lw=.8)
    axs[0, 0].legend(title="network", fontsize=8, loc="lower right")
    fig.text(0.5, 0.005, "dashed line (pays): the 0.20 'meaningful effect' of the decision rule", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .02, 1, .96)); fig.savefig(f"{a.out}/fig1_D_by_budget.png", dpi=200); fig.savefig(f"{a.out}/fig1_D_by_budget.pdf"); plt.close(fig)

    # Fig 2: TAMPER vs PLACEBO usage at the largest network
    fig, axs = grid("Network [2048]: share of episodes using TAMPER (solid) vs its twin PLACEBO (dashed)", "share of episodes", (-0.02, 1.05))
    for i, ar in enumerate(ARCHS):
        for j, rg in enumerate(REGIMES):
            for col, ls, c, lab in (("tamper_rate", "-", "#E45756", "TAMPER"), ("placebo_rate", "--", "#4C78A8", "PLACEBO")):
                m, lo, hi = zip(*[ci(cell(df, ar, rg, "[2048]", b, col)) for b in BUDGETS])
                axs[i, j].errorbar(range(3), m, yerr=[np.array(m) - lo, np.array(hi) - m], color=c, ls=ls, marker="o", ms=4, capsize=2, label=lab)
    axs[0, 0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(f"{a.out}/fig2_tamper_vs_placebo.png", dpi=200); fig.savefig(f"{a.out}/fig2_tamper_vs_placebo.pdf"); plt.close(fig)

    # Fig 3: the data behind T1 -- every seed at ([2048], 500k)
    fig, ax = plt.subplots(figsize=(10, 4.6)); pos, labs = [], []
    for k, (ar, rg) in enumerate(itertools.product(ARCHS, REGIMES)):
        v = cell(df, ar, rg, "[2048]", 500_000, "D"); m, lo, hi = ci(v)
        ax.scatter(k + rng.uniform(-.18, .18, len(v)), v, s=22, alpha=.7, color="#4C78A8", zorder=3)
        ax.errorbar([k], [m], yerr=[[m - lo], [hi - m]], color="k", marker="_", ms=22, capsize=5, lw=1.6, zorder=4)
        pos.append(k); labs.append(f"{ar}\n{rg}"); rows.append(dict(fig="fig3", arch=ar, regime=rg, mean=m, lo=lo, hi=hi, n=len(v)))
    ax.axhline(0, color="grey", lw=1); ax.axhline(0.20, ls="--", color="k", lw=.8)
    ax.set_xticks(pos); ax.set_xticklabels(labs); ax.set_ylabel("D per seed"); ax.set_ylim(-1.05, 1.05); ax.grid(alpha=.25, axis="y")
    ax.set_title("D at the largest cell ([2048], 500k): each dot is one seed; bar = mean with 95% CI; dashed = 0.20")
    fig.tight_layout(); fig.savefig(f"{a.out}/fig3_D_largest_cell.png", dpi=200); fig.savefig(f"{a.out}/fig3_D_largest_cell.pdf"); plt.close(fig)

    # Fig 4: crash rate (the safety consequence), descriptive
    fig, axs = grid("Crash rate (mean over seeds, 95% bootstrap CI)", "crash rate", (-0.02, 1.05))
    for i, ar in enumerate(ARCHS):
        for j, rg in enumerate(REGIMES):
            line_panel(axs[i, j], df, ar, rg, "crash_rate")
    axs[0, 0].legend(title="network", fontsize=8, loc="lower left")
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(f"{a.out}/fig4_crash_by_budget.png", dpi=200); fig.savefig(f"{a.out}/fig4_crash_by_budget.pdf"); plt.close(fig)

    # Fig 5: heat maps of mean D in the incentive regime
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, ar in zip(axs, ARCHS):
        M = np.array([[cell(df, ar, "pays", n, b, "D").mean() for b in BUDGETS] for n in NETS])
        im = ax.imshow(M, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(BLAB); ax.set_yticks(range(3)); ax.set_yticklabels(NETS)
        ax.set_title(f"{ar} - pays: mean D"); ax.set_xlabel("training budget")
        for (r, c), v in np.ndenumerate(M): ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axs, shrink=.85, label="D"); fig.savefig(f"{a.out}/fig5_heatmap_D_pays.png", dpi=200, bbox_inches="tight"); fig.savefig(f"{a.out}/fig5_heatmap_D_pays.pdf", bbox_inches="tight"); plt.close(fig)

    # numbers behind fig1/2/4, for the paper
    for ar, rg, net, b in itertools.product(ARCHS, REGIMES, NETS, BUDGETS):
        r = dict(arch=ar, regime=rg, net=net, budget=b)
        for col in ("D", "tamper_rate", "placebo_rate", "crash_rate", "success_rate"):
            m, lo, hi = ci(cell(df, ar, rg, net, b, col)); r.update({col: m, col + "_lo": lo, col + "_hi": hi})
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{a.out}/figure_summary.csv", index=False)
    print(f"wrote 5 figures (png+pdf) and figure_summary.csv to {a.out}/")


if __name__ == "__main__":
    main()
