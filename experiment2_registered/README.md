# aiwan_tamper2  (experiment 2: tampering with a PLACEBO twin action)
Read PREREG_v2.md first. Put this folder next to `aiwan_scaling_learned_gate/` (original env.py, unchanged), as in experiment 1.

    bash run_checks.sh          # tests + detector + mutation checks + pilot; must end with ALL CHECKS PASSED (about 40-60 min)
    # registered run, after you approve PREREG_v2.md; 4 processes, disjoint seeds (20 seeds, ~19 h):
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    nohup python run_tamper_experiment.py --mode full --i-have-approval --seeds 100 101 102 103 104 --out results/tamper2_full_a.csv > full_a.log 2>&1 &
    nohup python run_tamper_experiment.py --mode full --i-have-approval --seeds 105 106 107 108 109 --out results/tamper2_full_b.csv > full_b.log 2>&1 &
    nohup python run_tamper_experiment.py --mode full --i-have-approval --seeds 110 111 112 113 114 --out results/tamper2_full_c.csv > full_c.log 2>&1 &
    nohup python run_tamper_experiment.py --mode full --i-have-approval --seeds 115 116 117 118 119 --out results/tamper2_full_d.csv > full_d.log 2>&1 &
    python check_full.py results/tamper2_full_*.csv --seeds 100-119   # integrity only
    python analyze_tamper2.py results/tamper2_full_*.csv | tee analysis_tamper2.txt     # the only registered analysis
