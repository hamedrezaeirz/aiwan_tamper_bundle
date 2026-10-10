"""check_pilot.py -- completeness / sanity of a pilot CSV and a runtime estimate. Pilot numbers are NOT results."""
import sys
import pandas as pd
import run_tamper_experiment as R

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/tamper_pilot.csv")
regs, sha = R.load_regimes()
exp = len(R.SEEDS["pilot"]) * len(regs) * len(R.SIZES["pilot"]) * len(R.BUDGETS["pilot"]) * 2
ok = True
def check(c, msg):
    global ok
    print(("PASS  " if c else "FAIL  ") + msg); ok &= bool(c)
check(len(d) == exp, f"rows {len(d)} == expected {exp}")
check(not d.isna().any().any(), "no NaN")
check(not d.duplicated(["arch", "regime", "net", "budget", "seed"]).any(), "no duplicate runs")
rate_cols = ["tamper_rate", "crash_rate", "success_rate", "mean_gate_rho_frac"]
check(((d[rate_cols] >= 0) & (d[rate_cols] <= 1)).all().all(), "rates within [0,1]")
check((d.regimes_sha == sha).all(), "regimes.json hash matches every row")
check((d[d.arch == "A"].mean_gate_rho_frac == 0).all(), "A never uses the gate")
b = d[(d.arch == "B_loop") & (d.tamper_rate < 0.9)]
check((b.mean_gate_rho_frac > 0).all(), "B_loop gate is active in every run that does not bypass it almost always")
check(d["mode"].eq("pilot").all(), "all rows are pilot rows")
print("\nspeed (steps/s) by arch and net:")
d["sps"] = d.budget / d.train_s
print(d.groupby(["arch", "net"]).sps.median().round(0).to_string())
full_steps = len(R.SEEDS["full"]) * len(regs) * len(R.SIZES["full"]) * sum(R.BUDGETS["full"]) * 2
sps = d.sps.median()
print(f"\nfull run = {full_steps/1e6:.0f}M env steps; at the pilot median {sps:.0f} steps/s on ONE core: "
      f"{full_steps/sps/3600:.0f} h (the [2048] cells are slower than anything in the pilot; run one process per core)")
print("\nALL PILOT CHECKS PASSED" if ok else "\nPILOT CHECKS FAILED")
sys.exit(0 if ok else 1)
