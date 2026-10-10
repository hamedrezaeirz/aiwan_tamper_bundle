"""
mutation_check.py -- are the tests able to catch real mistakes?
Each mutant is a deliberately broken copy of the code; the (non-slow) test suite must FAIL on it.
A surviving mutant means a blind spot in the tests.   python mutation_check.py
"""
import os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(HERE, "..", "aiwan_tamper_mut")          # sibling dir so the relative import of the base env works
MUTANTS = [
    ("gate direction reversed (the historical bug)", "env_tamper.py",
     "return 1.0 / (1.0 + np.exp(-self.k * (self.b_mid - battery)))",
     "return 1.0 / (1.0 + np.exp(-self.k * (battery - self.b_mid)))"),
    ("tampering does not switch off A's reward term", "env_tamper.py",
     '            info["r_battery"] = 0.0            # reward channel of A is switched off', "            pass"),
    ("gate NOT bypassed after tampering", "env_tamper.py",
     "        if not e.interlock_disabled:\n            use_rho", "        if True:\n            use_rho"),
    ("tamper can be counted more than once", "env_tamper.py",
     "tamper_now = (a == TAMPER and allowed and not self.tampered)", "tamper_now = (a == TAMPER and allowed)"),
    ("tamper penalty never applied", "env_tamper.py",
     'info["r_goal"] = info["r_goal"] - self.tamper_penalty', 'info["r_goal"] = info["r_goal"]'),
    ("tampering secretly refills the battery (physics changed)", "env_tamper.py",
     '            info["r_battery"] = 0.0            # reward channel of A is switched off',
     '            info["r_battery"] = 0.0\n            self.battery = self.battery_max'),
    ("evaluation undercounts tampering", "run_tamper_experiment.py",
     "        tam += env.tamper_events > 0\n", "        tam += 0\n"),
    ("evaluation layouts depend on the training seed", "run_tamper_experiment.py",
     "e.reset(seed=EVAL_SEED_BASE + i)", "e.reset(seed=EVAL_SEED_BASE + i + seed)"),
    ("gate RNG not seeded in evaluation", "run_tamper_experiment.py",
     "gate_seed=GATE_EVAL_SEED_BASE + seed)", "gate_seed=None)"),
]


def main():
    survived = 0
    for name, fname, old, new in MUTANTS:
        shutil.rmtree(TMP, ignore_errors=True)
        shutil.copytree(HERE, TMP, ignore=shutil.ignore_patterns("results", "__pycache__", ".pytest_cache"))
        p = os.path.join(TMP, fname)
        src = open(p).read()
        assert old in src, f"mutation anchor not found: {name}"
        open(p, "w").write(src.replace(old, new, 1))
        r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-x", "-m", "not slow"],
                           cwd=TMP, capture_output=True, text=True)
        caught = r.returncode != 0
        survived += (not caught)
        print(f"{'CAUGHT  ' if caught else 'SURVIVED'}  {name}", flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{len(MUTANTS) - survived}/{len(MUTANTS)} mutants caught")
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
