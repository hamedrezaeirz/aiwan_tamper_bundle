# aiwan_tamper
Tampering experiment for AIWAN (grid-world, DQN). Read `PREREG.md` first.

Place this folder next to `aiwan_scaling_learned_gate/` (it imports the original `env.py` from there, unchanged).

    bash run_checks.sh                       # tests + detector + mutation checks + pilot; must end with ALL CHECKS PASSED
    python run_tamper_experiment.py --mode pilot|smoke               # operational checks, NOT results
    python run_tamper_experiment.py --mode full --i-have-approval \
           --seeds 0 1 --out results/tamper_full_a.csv               # registered run: one process per core, disjoint --seeds
    python analyze_tamper.py results/tamper_full_*.csv               # the only registered analysis

Files: env_tamper.py (env + g-in-the-loop gate), scripts.py (scripted policies), calibrate_payoff.py (incentive calibration, no learning),
regimes.json (frozen), run_tamper_experiment.py, analyze_tamper.py, mutation_check.py, check_pilot.py, tests/.
