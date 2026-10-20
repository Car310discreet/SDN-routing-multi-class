"""
controller_routing.py - multi-class ML path selection on GEANT2 through Floodlight's REST API.

    1. Top-5 candidate paths Sender -> Receiver by hop count: Yen's K shortest loopless paths
       (baseline/path_discovery.py), optionally cross-checked against Floodlight's own
       /wm/routing/paths API.
    2. Passive link utilisation from Floodlight port counters (once).
    3. For each candidate: pin its ICMP/TCP with static entries, measure the path features
       (telemetry.py), remove the entries, and predict its traffic class
       (Extremely Low / Low / High / Extremely High) with the trained model.
    4. Choose the candidate with the lowest predicted class. Ties go to the lower probability-weighted
       class, then to the shorter / earlier candidate.
    5. Push bidirectional static flow entries (StaticEntryPusher) for all Sender <-> Receiver traffic on it.

Probing needs the Mininet hosts, so routing runs inside run_experiment.py. Without root this script
prints the candidates and compares them with Floodlight's when the controller is reachable:
    .venv/bin/python geant2/controller_routing.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geant2_topo import RECEIVER, SENDER, build_geant2  # noqa: E402
from ml_classifier import CongestionClassifier  # noqa: E402
from path_discovery import discover_paths  # noqa: E402
from telemetry import TelemetryClient  # noqa: E402
from telemetry_and_model import ControllerError  # noqa: E402
from traffic_gen import CLASS_NAMES  # noqa: E402

log = logging.getLogger(__name__)
K_PATHS = 5
PROBE_PRIORITY = 40000
ROUTE_PRIORITY = 32768
ROUTE_PREFIX = "geant2ml"


def candidate_paths(spec, src_host: str = SENDER, dst_host: str = RECEIVER, k: int = K_PATHS) -> list[list[str]]:
    return discover_paths(spec.graph, spec.hosts[src_host].switch, spec.hosts[dst_host].switch, k).paths


def floodlight_paths(api, spec, src_switch: str, dst_switch: str, k: int = K_PATHS) -> list[list[str]]:
    """Floodlight's k shortest paths (/wm/routing/paths) as switch-name lists."""
    names = spec.dpid_to_switch
    reply = api._request("GET", f"/wm/routing/paths/{spec.switch_dpids[src_switch]}/"
                                f"{spec.switch_dpids[dst_switch]}/{k}/json")
    paths = []
    for result in reply.get("results", []):
        switches = []
        for hop in result.get("path", []):
            name = names.get(hop.get("dpid"))
            if name and (not switches or switches[-1] != name):
                switches.append(name)
        paths.append(switches)
    return paths


@dataclass
class ScoredPath:
    index: int
    switches: list[str]
    features: dict
    predicted_class: int
    predicted_name: str
    probabilities: list[float]
    expected_class: float
    probe_s: float
    inference_ms: float


@dataclass
class Decision:
    model: str
    chosen_index: int
    chosen: list[str]
    scored: list[ScoredPath] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def choose(scored: list[ScoredPath]) -> ScoredPath:
    return min(scored, key=lambda s: (s.predicted_class, round(s.expected_class, 6), len(s.switches), s.index))


def ml_routing(candidates: list[list[str]], src_host: str, dst_host: str, installer, telemetry,
               classifier: CongestionClassifier, route_priority: int = ROUTE_PRIORITY) -> tuple[Decision, list[str]]:
    """Classify every candidate and route src <-> dst over the least congested one."""
    if not candidates:
        raise ValueError("No candidate paths")
    spec = telemetry.spec
    src_ip, dst_ip = spec.hosts[src_host].ip, spec.hosts[dst_host].ip
    link_util = telemetry.link_utilization()
    scored = []
    for index, path in enumerate(candidates):
        names = installer.install_path(src_ip, dst_ip, path, prefix=f"g2probe{index}", priority=PROBE_PRIORITY,
                                       ip_protos=[1, 6])
        started = time.monotonic()
        try:
            features = telemetry.measure(src_host, dst_host, path, link_util)
        finally:
            installer.remove(names)
        predicted, probabilities, inference_ms = classifier.predict(features)
        scored.append(ScoredPath(index, path, {k: round(v, 4) for k, v in features.items()}, predicted,
                                 CLASS_NAMES[predicted], [round(float(p), 4) for p in probabilities],
                                 round(classifier.expected_class(probabilities), 4),
                                 round(time.monotonic() - started, 2), round(inference_ms, 4)))
        log.info("Path %d %s: %s (P=%s, rtt %.0f ms, %.2f Mbps)", index, "-".join(path), CLASS_NAMES[predicted],
                 ", ".join(f"{p:.2f}" for p in probabilities), features["rtt_avg_ms"], features["throughput_mbps"])
    best = choose(scored)
    installed = installer.install_path(src_ip, dst_ip, best.switches, prefix=ROUTE_PREFIX, priority=route_priority)
    return Decision(classifier.name, best.index, best.switches, scored), installed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    spec = build_geant2()
    ours = candidate_paths(spec)
    print(f"Top-{K_PATHS} hop-count candidates {SENDER} -> {RECEIVER} (Yen):")
    for i, path in enumerate(ours):
        print(f"  {i}: {'-'.join(path)} ({len(path) - 1} hops)")
    api = TelemetryClient(args.controller_url, retries=1)
    if not api.health():
        print("Floodlight not reachable; skipped the comparison with /wm/routing/paths")
        return
    try:
        theirs = floodlight_paths(api, spec, ours[0][0], ours[0][-1])
    except ControllerError as exc:
        print(f"Floodlight did not return paths ({exc}); is GEANT2 running?")
        return
    if not theirs:
        print("Floodlight returned no paths; is GEANT2 running?")
        return
    print("Floodlight /wm/routing/paths:")
    for i, path in enumerate(theirs):
        print(f"  {i}: {'-'.join(path)} ({len(path) - 1} hops){'' if path in ours else '  (not in Yen set)'}")


if __name__ == "__main__":
    main()
