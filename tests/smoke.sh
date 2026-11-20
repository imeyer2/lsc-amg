#!/bin/bash
# End-to-end smoke test of the full pipeline on a tiny problem, run locally
# (no SLURM): train 5 seeds x 2 epochs, select, benchmark, scale, compare,
# figures. Takes a few minutes on a multi-core node. Writes to outputs_smoke/
# by pointing OUTPUTS there, so it never touches a real reproduction.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export LSC_OUTPUTS="outputs_smoke"
rm -rf outputs_smoke
# Bit-identity checks (tests/test_problems.py, tests/test_twolevel.py --identity-only) compare hashes
# recorded on the reference platform (a Linux x86_64 cluster). Elsewhere floating-point last bits, and the
# perforated_l Triangle mesh, can differ, so they only warn unless SMOKE_STRICT=1.
ident() {
  if [ "${SMOKE_STRICT:-0}" = 1 ]; then "$@"
  else "$@" || echo "WARNING: '$*' differs from the reference-platform hashes (expected off the reference Linux x86_64 platform; SMOKE_STRICT=1 makes this fatal)"
  fi
}
ident python tests/test_problems.py
python tests/test_fast_policy.py
ident python tests/test_twolevel.py --identity-only
python tests/test_twolevel.py --pipeline-only
E="--experiments tests/smoke_experiments.csv --scalability tests/smoke_scalability.csv --local"
python scripts/reproduce.py train $E
python scripts/reproduce.py select $E
python scripts/reproduce.py benchmark $E
python scripts/reproduce.py scale $E
python scripts/reproduce.py status $E
python scripts/reproduce.py compare $E > /dev/null
python scripts/reproduce.py figures $E > outputs_smoke/figures.log 2>&1
python - <<'PY'
import json
b = json.load(open("outputs_smoke/best_runs.json"))["perforated_l"]["smoke"]
assert b["complete"], b
r = json.load(open(f"{b['out_dir']}/preconditioner_benchmark/benchmark_results.json"))
assert all("rho_vcycle" in r[k]["schemes"]["learned"] for k in r), "rho_vcycle missing"
assert len(r) == 32
s = json.load(open("outputs_smoke/scalability_perforated_l_smokebig/preconditioner_benchmark/benchmark_results.json"))
assert len(s) == 16
print("SMOKE TEST PASSED: best seed", b["best_seed"], "R", round(b["R"], 4))
PY
