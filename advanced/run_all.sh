#!/usr/bin/env bash
# run_all.sh - train -> offline timing benchmark -> live Mininet benchmark -> figures.
# Floodlight must already be running (./start_floodlight.sh in another terminal).
#
#     advanced/run_all.sh                 # 3 scenarios, default hop-count path forced to Extremely High
#     SCENARIOS=5 advanced/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SCENARIOS=${SCENARIOS:-3}

curl -sf http://127.0.0.1:8080/wm/core/health/json >/dev/null \
    || { echo "Floodlight is not reachable on :8080 - start ./start_floodlight.sh first" >&2; exit 1; }

[ -f advanced/models/multiclass_bundle.joblib ] || "$PY" advanced/train_models.py
"$PY" advanced/run_experiment.py --benchmark-only

# systemd-inhibit keeps the laptop awake: a suspend mid-run corrupts every timing.
sudo systemd-inhibit --what=sleep:idle --why="GEANT2 benchmark" \
    "$PY" advanced/run_experiment.py --scenarios "$SCENARIOS" --congest-default-path

run=$(ls -td results/advanced/advanced_* | head -1)
"$PY" advanced/visualize_benchmarks.py "$run" --benchmark results/advanced/decision_latency_benchmark.csv
echo "figures: $run/figures"
