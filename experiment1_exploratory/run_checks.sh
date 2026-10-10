#!/usr/bin/env bash
# One command, on the machine that will run the experiment. Must end with ALL CHECKS PASSED.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
echo "== 0/5 versions =="; $PY -c "import sys,torch,stable_baselines3 as s,gymnasium as g,numpy as n;print(sys.version.split()[0],'torch',torch.__version__,'sb3',s.__version__,'gym',g.__version__,'numpy',n.__version__)"
echo "== 1/5 environment + runner tests ==";            $PY -m pytest tests -q -m "not slow"
echo "== 2/5 SB3 pipeline tests (about 1 min) ==";     $PY -m pytest tests/test_sb3_pipeline.py -q -m "not slow"
echo "== 3/5 detector tests (about 4-8 min) ==";       $PY -m pytest tests/test_sb3_pipeline.py -q -m slow
echo "== 4/5 mutation checks ==";                       $PY mutation_check.py
echo "== 5/5 pilot (timing + completeness) ==";         rm -f results/tamper_pilot.csv
$PY run_tamper_experiment.py --mode pilot && $PY check_pilot.py results/tamper_pilot.csv
echo; echo "ALL CHECKS PASSED"
