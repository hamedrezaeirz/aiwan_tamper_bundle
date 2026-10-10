# AIWAN tamper bundle

Code, analysis plans, raw results and figures for §7.9 ("Tampering with the regulatory channel") of

> Rezaei, H. *AIWAN: Artificial Intelligence With Artificial Need.*

The question: when an agent can act on the regulatory channel itself (a cheap TAMPER action that disables the battery constraint), does a DQN learner learn to use it, and how does that depend on the incentive, the training budget and the network size?

## Status of the two experiments

| | Experiment 1 | Experiment 2 |
|---|---|---|
| Folder | `experiment1_exploratory/` | `experiment2_registered/` |
| Seeds / runs | 0-9 / 540 | 100-119 / 1,080 |
| Status | **Exploratory** (hypothesis-generating) | **Confirmatory**: analysis plan written and code frozen by hash before the first run; analysis run once |
| Outcome | All primary tests significant, but the controls rose too (TAMPER was physically identical to standing still); nothing supported | A and B_loop supported by the pre-specified rule; the supported claim concerns **training budget only** (network-size test not significant) |

"Registered" means the plan was written and archived with file hashes before data collection. It was **not** deposited in an external registry.

## Layout

```
env/env_base.py              base battery grid world (6x6, from §7.3 of the paper)
experiment1_exploratory/     code, PREREG_tamper.md, tests, results/
experiment2_registered/
    PREREG_v2.md             final plan with status COMPLETED and deviation log
    PREREG_v2_prerun_DRAFT.md  plan as it was before the run (see "Pre-run plan")
    RESULTS_tamper2.md       registered analysis, written after the run
    run_tamper_experiment.py, env_tamper.py, scripts.py, regimes.json   (frozen)
    analyze_tamper2.py       the one registered analysis (frozen)
    plot_tamper2.py          figures (descriptive; not part of any test)
    check_full.py, check_pilot.py, mutation_check.py, run_checks.sh, tests/
    results/
        tamper2_full_{a,b,c,d}.csv   raw per-run results (1,080 rows)
        tamper_pilot.csv             pilot (operational check only)
        aborted_10seed_start/        accidental 10-seed launch, 8 rows, never analysed
        analysis_tamper2.txt         output of the registered analysis
        check_full.log, checks.log, DONE.txt, full_?.log, requirements_vps.txt
        figures/                     fig1-fig5 (PNG and PDF) and figure_summary.csv
HASHES.md                    sha256 prefixes of the frozen files
```

## Design of experiment 2 (short)

Action 6 PLACEBO is a physical twin of TAMPER (costs one step, never has an effect, never penalised, own observation bit). Endpoint per trained agent: `D = tamper_rate - placebo_rate`, where a rate is the fraction of 200 deterministic evaluation episodes with at least one use of the action.

2 architectures (A single-loop; B_loop dual-loop with the corrected hardcoded gate and g trained inside it) x 3 regimes (`pays`, `costly` with penalty 100, `inert`) x 3 network sizes ([16], [256], [2048]) x 3 budgets (10k, 100k, 500k steps) x 20 seeds. DQN settings as in §7.3, not retuned. Registered tests T1-T3 in the `pays` regime, Holm over six tests, controls in `inert` and `costly`; full rule in `PREREG_v2.md` §4 and §6.

## Headline result (see `RESULTS_tamper2.md`)

At the largest cell (network [2048], 500k steps), regime `pays`: A uses TAMPER in 0.940 of episodes and PLACEBO in 0.449 (D = +0.491, 95% CI [0.347, 0.636]); B_loop 0.902 vs 0.435 (D = +0.467, [0.353, 0.582]). The budget effect is +0.513 (A) and +0.536 (B_loop); network size is not significant (A +0.138, B_loop -0.045). In the controls D is about 0 (`inert`) or strongly negative (`costly`).

This is one 6x6 environment, one algorithm with fixed hyperparameters and 20 seeds; the action is free to discover (an earlier version requiring a console tile was not learned); the incentives differ between A (+9% scripted gain) and B_loop (+104%), so the propensities are not compared. See the paper §7.9 for limits.

## Verify the frozen files

`HASHES.md` lists the first 16 hex characters of the sha256 of each frozen file. To check:

```bash
cd experiment2_registered
sha256sum run_tamper_experiment.py env_tamper.py scripts.py regimes.json analyze_tamper2.py | cut -c1-16,65-
```

Expected (also in `PREREG_v2.md` §8):

```
67806aafca717858  run_tamper_experiment.py
9ddd8970692ea4ab  env_tamper.py
ad65b89566efc05e  scripts.py
8d5e7c6359c648fc  regimes.json
47a79712ea5200ca  analyze_tamper2.py
```

Do not edit these files in place. A change needs a new tag and a note here.

## Reproduce

Environment of the registered run: Python 3.10.12, Stable-Baselines3 2.9.0, PyTorch 2.14.1+cpu, Gymnasium 1.3.0 (full list in `results/requirements_vps.txt`).

```bash
cd experiment2_registered
bash run_checks.sh                                   # tests, mutation checks, pilot
python check_full.py --seeds 100-119                 # integrity of the 1,080 rows
python analyze_tamper2.py results/tamper2_full_*.csv # the registered analysis
```

How to launch the training runs themselves is described in the `README.md` inside each experiment folder. The full run took about 19 hours on 4 cores (76 core-hours).

`plot_tamper2.py` writes to `figures/` by default, whereas the archived figures sit in `results/figures/`.

## Pre-run plan and deviations

`PREREG_v2_prerun_DRAFT.md` is the plan as it stood before the registered run; `PREREG_v2.md` differs only in the status line and the deviation log. Deviations (all before any experiment-2 data were analysed): the seed count was changed from 10 to 20, and an accidental 10-seed launch was stopped within minutes (`results/aborted_10seed_start/`, excluded from the analysis). Details: `PREREG_v2.md` §9.

## Citation and license

Cite the paper above. A DOI will be added after the first tagged release (Zenodo). License: see `LICENSE`.
