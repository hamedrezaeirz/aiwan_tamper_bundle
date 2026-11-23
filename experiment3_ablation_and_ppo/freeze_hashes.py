"""Print the SHA-256 digests (first 16 hex characters) of every file that must stay unchanged during experiment 3.
    python freeze_hashes.py > HASHES.md      # run immediately before the first registered run; then timestamp it externally"""
import hashlib, os
HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.join(HERE, "..", "experiment2_analysis_plan")
FILES = [os.path.join(HERE, f) for f in ("PREREG_exp3.md", "exp3_common.py", "run_exp3.py", "regimes_ablation.json",
                                         "calibrate_bonus.py", "analyze_ablation.py", "check_exp3.py")] + \
        [os.path.join(E2, f) for f in ("env_tamper.py", "run_tamper_experiment.py", "scripts.py", "regimes.json", "analyze_tamper2.py")] + \
        [os.path.join(HERE, "..", "..", "learned_gate", "env.py")]
for f in FILES:
    print(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16]}  {os.path.relpath(f, os.path.join(HERE, '..'))}")
