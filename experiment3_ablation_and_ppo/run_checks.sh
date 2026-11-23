#!/bin/bash
# All fast + slow checks of experiment 3. Does not touch any registered result.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
python -m pytest -q tests/test_exp3.py tests/test_analyze_ablation.py
(cd ../experiment2_analysis_plan && python -m pytest -q tests/test_env_tamper.py tests/test_runner_fake.py)
python calibrate_bonus.py | tail -2
echo "ALL CHECKS PASSED"
