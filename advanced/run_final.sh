#!/usr/bin/env bash
# run_final.sh - every result for the GEANT2 vs fat-tree comparison, into final_results/.
#   1. train each topology's ensemble if its bundle is missing
#   2. offline compute benchmark on both topologies (Stage 1 per method, batched vs per-candidate inference)
#   3. live Mininet benchmark on GEANT2, then on the fat-tree: hop-count, latency, bandwidth, hop-count + ML
#      (baseline) and combined + ML (ours); the hop-count shortest path is forced to Extremely High
#   4. advanced/final_results.py over the two new runs -> final_results/
# Floodlight must already be running (./start_floodlight.sh). Both live runs share one sudo call, so the
# password is asked once, up front, and the laptop is kept awake until they finish.
#
#     advanced/run_final.sh                 # 5 scenarios per topology, about 2-2.5 hours
#     SCENARIOS=3 advanced/run_final.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SCENARIOS=${SCENARIOS:-5}
OUT=results/advanced

curl -sf http://127.0.0.1:8080/wm/core/health/json >/dev/null \
    || { echo "Floodlight is not reachable on :8080 - start ./start_floodlight.sh first" >&2; exit 1; }

[ -f advanced/models/multiclass_bundle.joblib ] || "$PY" advanced/train_models.py --topology geant2
[ -f advanced/models/fattree_bundle.joblib ] || "$PY" advanced/train_models.py --topology fattree
for topology in geant2 fattree; do
    "$PY" advanced/run_experiment.py --benchmark-only --topology "$topology"
done

sudo -v
sudo systemd-inhibit --what=sleep:idle:handle-lid-switch --who="final results" --why="Mininet benchmark" \
    bash -c "
        set -e
        for topology in geant2 fattree; do
            mn -c >/dev/null 2>&1 || true
            '$PY' advanced/run_experiment.py --topology \"\$topology\" --scenarios '$SCENARIOS' --congest-default-path
        done
        mn -c >/dev/null 2>&1 || true"

geant2_run=$(ls -td "$OUT"/geant2_* | head -1)
fattree_run=$(ls -td "$OUT"/fattree_* | head -1)
"$PY" advanced/final_results.py --geant2-run "$geant2_run" --fattree-run "$fattree_run"
echo "final results: final_results/  (runs: $geant2_run, $fattree_run)"
