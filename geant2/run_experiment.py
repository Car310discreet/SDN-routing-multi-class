"""
run_experiment.py - GEANT2: multi-class ML routing vs the baseline approaches (needs root).

Approaches, each run under the same background-traffic scenario:
  hop_count   Floodlight's default reactive routing (hop-count metric)
  qos_aware   Floodlight's reactive routing with the latency metric
  airsdn      the baseline paper's approach, unchanged: probe the 5 hop-count candidates (19 features:
              ping, iperf3, NFStream), binary Logistic Regression (models/logistic_regression_model.pkl),
              lowest predicted level wins
  geant2_ml   this work: the same 5 candidates, 11 path features, the best 4-class model
              (geant2/models/best_model.joblib), lowest predicted class wins

For every scenario (`--scenarios N`, seeds `--seed`, `--seed + 1`, ...) the traffic generator applies one
fixed set of link loads. Each approach then chooses a route, and on that route (multimedia_evaluator.py):
  60 s RTT + ping packet loss, 60 s throughput (iperf3), and the 66 s H.264 video streamed
  Sender -> Receiver with FFmpeg, scored with PSNR and SSIM (video_quality.py: native H.264 decoder,
  so badly damaged video is scored too).

Results: results/geant2/geant2_<timestamp>/scenario_<k>/<approach>/
    decision.json (candidates, ground-truth class of each, predictions, chosen path),
    timeseries.csv, frames.csv, summary.json
and scenario_<k>/scenario.json (link loads).

    ./start_floodlight.sh                                          # separate terminal
    sudo .venv/bin/python geant2/run_experiment.py --scenarios 3
    .venv/bin/python geant2/evaluate.py results/geant2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controller_routing import candidate_paths, floodlight_paths, ml_routing  # noqa: E402
from geant2_topo import RECEIVER, SENDER, build_geant2, chown_to_invoking_user, start_geant2  # noqa: E402
from ml_classifier import DEFAULT_MODEL, CongestionClassifier  # noqa: E402
from multimedia_evaluator import DEFAULT_VIDEO  # noqa: E402
from telemetry import PathTelemetry, TelemetryClient  # noqa: E402
from telemetry_and_model import (ControllerError, FlowInstaller, PathProber, TrafficClassifier,  # noqa: E402
                                 routing_algorithm)
from topology_builder import stop_network  # noqa: E402
from traffic_gen import CLASS_NAMES, TrafficGenerator, sample_scenario  # noqa: E402
from video_quality import Evaluator  # noqa: E402  (scores damaged video too)

log = logging.getLogger("run_experiment")
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPROACHES = ["hop_count", "qos_aware", "airsdn", "geant2_ml"]
FLOODLIGHT_METRIC = {"hop_count": "hopcount", "qos_aware": "latency", "airsdn": "hopcount", "geant2_ml": "hopcount"}


def write_json(path: str, data) -> None:
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, default=str)


def ground_truth(scenario, path: list[str] | None) -> dict | None:
    if not path:
        return None
    return {"bottleneck_load": round(scenario.path_utilization(path), 4),
            "class": scenario.path_class(path), "class_name": CLASS_NAMES[scenario.path_class(path)]}


def route(approach, api, spec, net, installer, scenario, candidates, args, models) -> tuple[dict, list[str]]:
    """Let `approach` choose the Sender -> Receiver route; returns its decision record and static entries."""
    decision = {"approach": approach, "floodlight_metric": FLOODLIGHT_METRIC[approach],
                "candidates": [{"index": i, "path": p, "ground_truth": ground_truth(scenario, p)}
                               for i, p in enumerate(candidates)]}
    installed: list[str] = []
    if approach in ("hop_count", "qos_aware"):
        try:                              # the path Floodlight ranks first under the current metric
            first = floodlight_paths(api, spec, candidates[0][0], candidates[0][-1], 1)
            decision["chosen_path"] = first[0] if first else None
        except ControllerError as exc:
            log.warning("Could not read Floodlight's route: %s", exc)
            decision["chosen_path"] = None
    elif approach == "airsdn":
        # the baseline routes from its video client (the Receiver) to its server (the Sender)
        prober = PathProber(net, spec, probe_seconds=args.probe_seconds)
        selection, installed = routing_algorithm([p[::-1] for p in candidates], RECEIVER, SENDER, installer,
                                                 prober, models["airsdn"])
        decision.update(stage2=selection.to_dict(), chosen_path=selection.chosen[::-1])
    else:
        telemetry = PathTelemetry(net, spec, api)
        selection, installed = ml_routing(candidates, SENDER, RECEIVER, installer, telemetry, models["geant2_ml"])
        decision.update(stage2=selection.to_dict(), chosen_path=selection.chosen)
    decision["chosen_ground_truth"] = ground_truth(scenario, decision.get("chosen_path"))
    return decision, installed


def run(args) -> str:
    if os.geteuid() != 0:
        raise SystemExit("Mininet needs root: run with sudo")
    api = TelemetryClient(args.controller_url)
    if not api.health():
        raise SystemExit(f"Floodlight is not reachable at {args.controller_url} (start ./start_floodlight.sh)")
    models = {}
    if "airsdn" in args.approaches:
        models["airsdn"] = TrafficClassifier(args.baseline_model)
    if "geant2_ml" in args.approaches:
        models["geant2_ml"] = CongestionClassifier.load(args.model)
    spec = build_geant2(args.hosts_per_switch)
    candidates = candidate_paths(spec)
    run_dir = os.path.join(args.out, f"geant2_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    metadata = {"topology": "geant2", "sender": SENDER, "receiver": RECEIVER, "candidates": candidates,
                "geant2_model": getattr(models.get("geant2_ml"), "name", None), "scenarios": {}}

    net = generator = None
    installed: list[str] = []
    installer = FlowInstaller(api, spec)
    try:
        net = start_geant2(spec, api)
        generator = TrafficGenerator(net, spec, installer)
        generator.start()
        evaluator = Evaluator(net, spec, SENDER, RECEIVER, args.video)
        for k in range(args.scenarios):
            scenario = sample_scenario(spec.graph, args.seed + k, args.path_hops)
            generator.apply(scenario)
            time.sleep(args.warmup)
            scenario_dir = os.path.join(run_dir, f"scenario_{k}")
            os.makedirs(scenario_dir, exist_ok=True)
            write_json(os.path.join(scenario_dir, "scenario.json"), scenario.to_dict())
            metadata["scenarios"][k] = {"seed": scenario.seed, "approaches": {}}
            for approach in args.approaches:
                log.info("=== scenario %d: %s ===", k, approach)
                installer.remove(installed)
                installed = []
                api.set_routing_metric(FLOODLIGHT_METRIC[approach])
                time.sleep(6)            # reactive entries of the previous approach idle out (5 s timeout)
                out_dir = os.path.join(scenario_dir, approach)
                os.makedirs(out_dir, exist_ok=True)
                decision, installed = route(approach, api, spec, net, installer, scenario, candidates, args, models)
                log.info("%s routes over %s (ground truth: %s)", approach, "-".join(decision.get("chosen_path") or []),
                         (decision["chosen_ground_truth"] or {}).get("class_name"))
                write_json(os.path.join(out_dir, "decision.json"), decision)
                summary = evaluator.run(approach, out_dir)
                metadata["scenarios"][k]["approaches"][approach] = {
                    "summary": summary.__dict__, "chosen_path": decision.get("chosen_path"),
                    "chosen_ground_truth": decision["chosen_ground_truth"]}
                write_json(os.path.join(run_dir, "run.json"), metadata)
    finally:
        installer.remove(installed)
        if generator is not None:
            generator.stop()
        if net is not None:
            try:
                api.clear_flows()
            except ControllerError as exc:
                log.warning("Could not clear static entries: %s", exc)
        stop_network(net)
        write_json(os.path.join(run_dir, "run.json"), metadata)
        chown_to_invoking_user(args.out)
    log.info("Results in %s", run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--approaches", nargs="+", default=APPROACHES, choices=APPROACHES)
    parser.add_argument("--scenarios", type=int, default=1, help="traffic scenarios to evaluate")
    parser.add_argument("--seed", type=int, default=1000, help="seed of the first scenario")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="4-class model from geant2/train_models.py")
    parser.add_argument("--baseline-model", default=os.path.join(PROJECT_DIR, "models", "logistic_regression_model.pkl"))
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--probe-seconds", type=int, default=30, help="AIRSDN iperf3 probe length (its training data: 30)")
    parser.add_argument("--warmup", type=float, default=10.0, help="seconds after applying a scenario")
    parser.add_argument("--path-hops", type=float, default=6.0)
    parser.add_argument("--hosts-per-switch", type=int, default=1)
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out", default=os.path.join(PROJECT_DIR, "results", "geant2"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mininet").propagate = False
    try:
        run(args)
    except (ControllerError, FileNotFoundError, ValueError, TypeError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
