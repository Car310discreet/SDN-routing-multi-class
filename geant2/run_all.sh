#!/usr/bin/env bash
# run_all.sh - the whole GEANT2 pipeline in one command (Floodlight must be running).
#
#   1. dataset   collect_dataset.py   SAMPLES rows (default 4000, balanced; resumable)    root
#   2. models    train_models.py      top-4 models compared, best saved                  your user
#   3. routing   run_experiment.py    SCENARIOS scenarios (default 3): 4 approaches,     root
#                                     each streams the video Sender h15 -> Receiver h23
#   4. results   evaluate.py          tables and charts                                  your user
#
#   ./start_floodlight.sh                                   # separate terminal
#   sudo geant2/run_all.sh                                  # full run, about 12-13 hours
#   sudo SAMPLES=200 SCENARIOS=1 geant2/run_all.sh          # smoke test, about 1 hour
#
# Everything is logged to results/geant2/run_all.log. If it stops, rerun it: the dataset resumes
# where it left off.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
SAMPLES=${SAMPLES:-4000}
SCENARIOS=${SCENARIOS:-3}
LOG=results/geant2/run_all.log

[ "$(id -u)" -eq 0 ] || { echo "Mininet needs root: sudo $0" >&2; exit 1; }
curl -sf http://127.0.0.1:8080/wm/core/health/json >/dev/null \
    || { echo "Floodlight is not running: start ./start_floodlight.sh first" >&2; exit 1; }

as_user() { if [ -n "${SUDO_USER:-}" ]; then sudo -u "$SUDO_USER" "$@"; else "$@"; fi; }
hand_back() { [ -n "${SUDO_USER:-}" ] && chown -R "$SUDO_USER": results/geant2 geant2/data 2>/dev/null || true; }
trap hand_back EXIT

mkdir -p results/geant2
$PY -c "from mininet.clean import cleanup; cleanup()" >/dev/null 2>&1 || true   # leftovers of a killed run

{
    echo "== $(date '+%F %T') 1/4 dataset: $SAMPLES rows"
    $PY geant2/collect_dataset.py --samples "$SAMPLES"
    hand_back
    echo "== $(date '+%F %T') 2/4 training the top-4 models"
    as_user $PY geant2/train_models.py
    echo "== $(date '+%F %T') 3/4 routing and video transfer: $SCENARIOS scenario(s)"
    $PY geant2/run_experiment.py --scenarios "$SCENARIOS"
    hand_back
    echo "== $(date '+%F %T') 4/4 evaluation"
    as_user $PY geant2/evaluate.py results/geant2
    echo "== $(date '+%F %T') done"
} 2>&1 | tee -a "$LOG"
