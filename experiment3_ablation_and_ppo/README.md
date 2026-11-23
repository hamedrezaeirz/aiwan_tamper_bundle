# tamper / experiment 3: causal ablation (DQN) and PPO replication
The design and analysis plan are in `PREREG_exp3.md`. The folder sits next to `experiment2_analysis_plan/` and imports its frozen files (nothing there is modified).

    bash run_checks.sh              # all tests (about 1 minute), must end ALL CHECKS PASSED
    python verify_repro.py          # ON THE TARGET MACHINE: does retraining reproduce experiment-2 rows exactly? (about 2 minutes)
    python calibrate_bonus.py       # scripted choice of the bonus (b = 55), no learning
    python freeze_hashes.py > HASHES.md    # right before the first registered run; timestamp externally (Zenodo)
    # Part A and Part B commands: PREREG_exp3.md sections A6 and B4

| file | role |
|---|---|
| `exp3_common.py` | algorithm switch (dqn/ppo), `CRNInterlockedGEnv` (gate with common random numbers), `evaluate_paired` (real vs counterfactual replay) |
| `run_exp3.py` | runner; refuses `--mode full` without approval, without an explicit design, with pilot seeds or experiment-2 seeds |
| `regimes_ablation.json` | `pays` (as experiment 2) and `bonus` (TAMPER pays +55 once, bypasses nothing) |
| `analyze_ablation.py` | registered analysis of Part A (imports the statistics of `analyze_tamper2.py`) |
| `check_exp3.py` | integrity of finished CSVs (design complete, versions, seeds, real == counterfactual where it must be) |
| `results/` | registered runs: per-process CSVs and logs (`ablation_dqn_{200,205,210,215}.csv`, `ablation_*.log`; `ppo_{300,305,310,315}.csv`, `ppo_*.log`), integrity checks (`check_ablation.txt`, `check_ppo.txt`), registered analyses (`analysis_ablation.txt`, `analysis_ppo.txt`), `merged/` (the CSVs concatenated); and the pilot runs (seeds 900/901, NOT results) and the bonus calibration output. The saved models named in the `model_path` column are not included |

To re-check the reported results from the archived CSVs (about 1 minute; the output must match `results/check_*.txt` and `results/analysis_*.txt`):

    python check_exp3.py results/ablation_dqn_*.csv --algo dqn --archs B_loop --regimes pays bonus --sizes 16 2048 --budgets 10000 100000 500000 --seeds 200-219
    python analyze_ablation.py results/ablation_dqn_*.csv --seeds 200-219
    python check_exp3.py results/ppo_*.csv --algo ppo --archs A B_loop --regimes pays costly inert --sizes 16 2048 --budgets 10000 100000 500000 --seeds 300-319
    python ../experiment2_analysis_plan/analyze_tamper2.py results/ppo_*.csv

Part B needs no new analysis script: it uses `../experiment2_analysis_plan/analyze_tamper2.py` unchanged.
Pilot/smoke runs write `mode` = pilot/smoke and are rejected by every registered analysis.
