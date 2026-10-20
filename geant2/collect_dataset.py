"""
collect_dataset.py - build the 4-class GEANT2 path dataset (needs root and a running Floodlight).

Each epoch:
  1. the traffic generator draws and applies a new scenario (per-link offered loads), then waits
     `--warmup` seconds. With probability `--targeted` (default 0.7) the scenario targets the class
     that is furthest behind its quota (see traffic_gen.targeted_weights), so all four classes fill at
     a similar pace; otherwise it is an untargeted scenario;
  2. the passive link utilisation is read from Floodlight once;
  3. paths are probed one at a time (their ICMP/TCP pinned with static entries): the 5 hop-count
     candidates Sender -> Receiver, plus one random candidate of `--extra-pairs` random host pairs
     (more path lengths and links in the data);
  4. each path is labelled with the class of its bottleneck offered load (ground truth from the
     generator; no measured feature is used for the label).

Rows are appended to the CSV as they are measured, so a run can be stopped and resumed. A row whose
path is below 80 % load but lost every ping is a measurement failure (the controller lost the path), not
congestion, and is not written. An epoch whose Floodlight requests fail is skipped; three failed epochs
in a row stop the run. By default,
paths whose class already has `--samples / 4` rows are not probed, so the dataset ends up balanced.

The default of 4,000 rows (1,000 per class) is about 360 rows per feature for 11 features, enough
for the tree ensembles and the SVM to train and still leave 600 rows for validation and test. At
about 10 s per probe it takes 11-12 hours.

    ./start_floodlight.sh                                     # separate terminal
    sudo .venv/bin/python geant2/collect_dataset.py           # -> geant2/data/geant2_dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
import time
from collections import Counter

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geant2_topo import RECEIVER, SENDER, build_geant2, chown_to_invoking_user, hosts_on, start_geant2  # noqa: E402
from path_discovery import yen_hop_count  # noqa: E402
from telemetry import FEATURE_NAMES, PathTelemetry, TelemetryClient  # noqa: E402
from telemetry_and_model import ControllerError, FlowInstaller  # noqa: E402
from topology_builder import stop_network  # noqa: E402
from traffic_gen import CLASS_BANDS, CLASS_NAMES, TrafficGenerator, sample_scenario  # noqa: E402

log = logging.getLogger("collect_dataset")
GEANT2_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(GEANT2_DIR, "data", "geant2_dataset.csv")
META_COLUMNS = ["epoch", "time", "src", "dst", "path", "bottleneck_load"]
LABEL_COLUMNS = ["traffic_class", "traffic_class_name"]
PROBE_PRIORITY = 40000


def existing_rows(path: str) -> tuple[Counter, int]:
    """Class counts and next epoch number of an existing dataset (for resuming)."""
    counts, epoch = Counter(), 0
    if os.path.exists(path):
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                counts[int(row["traffic_class"])] += 1
                epoch = max(epoch, int(row["epoch"]) + 1)
    return counts, epoch


def epoch_paths(spec, rng: random.Random, candidates: list[list[str]], extra_pairs: int):
    """(src host, dst host, switch path) to probe this epoch."""
    probes = [(SENDER, RECEIVER, path) for path in candidates]
    switches = sorted(spec.graph.nodes, key=lambda s: int(s[1:]))
    while len(probes) < len(candidates) + extra_pairs:
        a, b = rng.sample(switches, 2)
        if nx.shortest_path_length(spec.graph, a, b) < 2:
            continue                       # adjacent switches: a single link, already pinned by the generator
        probes.append((hosts_on(spec, a)[0], hosts_on(spec, b)[0], rng.choice(yen_hop_count(spec.graph, a, b, 5))))
    return probes


def measure_epoch(spec, rng, candidates, args, scenario, counts, quota, installer, telemetry) -> list:
    """Probe this epoch's paths; returns (src, dst, path, features, label) rows that passed the checks."""
    rows, pending = [], Counter()
    link_util = telemetry.link_utilization()
    for index, (src, dst, path) in enumerate(epoch_paths(spec, rng, candidates, args.extra_pairs)):
        label = scenario.path_class(path)
        if args.balance and counts[label] + pending[label] >= quota:
            continue
        names = installer.install_path(spec.hosts[src].ip, spec.hosts[dst].ip, path,
                                       prefix=f"g2probe{index}", priority=PROBE_PRIORITY, ip_protos=[1, 6])
        try:
            features = telemetry.measure(src, dst, path, link_util)
        finally:
            installer.remove(names)
        if features["packet_loss_pct"] >= 100.0 and scenario.path_utilization(path) < CLASS_BANDS[-1][0]:
            log.warning("Discarded %s: every ping lost on a path below %.0f %% load (measurement failure)",
                        "-".join(path), 100 * CLASS_BANDS[-1][0])
            continue
        rows.append((src, dst, path, features, label))
        pending[label] += 1
    return rows


def collect(args) -> str:
    if os.geteuid() != 0:
        raise SystemExit("Mininet needs root: run with sudo")
    api = TelemetryClient(args.controller_url)
    if not api.health():
        raise SystemExit(f"Floodlight is not reachable at {args.controller_url} (start ./start_floodlight.sh)")
    spec = build_geant2(args.hosts_per_switch)
    candidates = yen_hop_count(spec.graph, spec.hosts[SENDER].switch, spec.hosts[RECEIVER].switch, 5)
    quota = math.ceil(args.samples / len(CLASS_NAMES))
    counts, first_epoch = existing_rows(args.out)
    log.info("Target %d rows (%s); already have %s", args.samples, "balanced" if args.balance else "unbalanced",
             dict(sorted(counts.items())))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    rng = random.Random(args.seed + first_epoch)

    def done() -> bool:
        if args.balance:
            return all(counts[c] >= quota for c in range(len(CLASS_NAMES)))
        return sum(counts.values()) >= args.samples

    net = generator = None
    installer = FlowInstaller(api, spec)
    try:
        net = start_geant2(spec, api)
        generator = TrafficGenerator(net, spec, installer)
        generator.start()
        telemetry = PathTelemetry(net, spec, api, probe_seconds=args.probe_seconds)
        with open(args.out, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=META_COLUMNS + FEATURE_NAMES + LABEL_COLUMNS)
            if new_file:
                writer.writeheader()
            epoch, failures = first_epoch, 0
            while not done() and (args.max_epochs is None or epoch < first_epoch + args.max_epochs):
                target = None
                if args.balance and rng.random() < args.targeted:
                    target = min(range(len(CLASS_NAMES)), key=lambda c: (counts[c] / quota, c))
                scenario = sample_scenario(spec.graph, rng.randrange(2 ** 31), args.path_hops, target)
                generator.apply(scenario)
                time.sleep(args.warmup)
                try:
                    measured = measure_epoch(spec, rng, candidates, args, scenario, counts, quota, installer, telemetry)
                    failures = 0
                except ControllerError as exc:
                    failures += 1
                    if failures >= 3:
                        raise
                    log.warning("Epoch %d skipped (%s); retrying in 30 s", epoch, exc)
                    time.sleep(30)
                    continue
                for src, dst, path, features, label in measured:
                    writer.writerow({"epoch": epoch, "time": round(time.time(), 1), "src": src, "dst": dst,
                                     "path": "-".join(path), "bottleneck_load": round(scenario.path_utilization(path), 4),
                                     **{k: round(v, 4) for k, v in features.items()},
                                     "traffic_class": label, "traffic_class_name": CLASS_NAMES[label]})
                    handle.flush()
                    counts[label] += 1
                log.info("Epoch %d done; rows per class: %s", epoch,
                         ", ".join(f"{CLASS_NAMES[c]} {counts[c]}" for c in range(len(CLASS_NAMES))))
                epoch += 1
    except KeyboardInterrupt:
        log.info("Stopped; rerun the same command to resume")
    finally:
        if generator is not None:
            generator.stop()
        if net is not None:
            try:
                api.clear_flows()
            except ControllerError as exc:
                log.warning("Could not clear static entries: %s", exc)
        stop_network(net)
        chown_to_invoking_user(os.path.dirname(os.path.abspath(args.out)))
    log.info("Dataset: %s (%d rows)", args.out, sum(counts.values()))
    return args.out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=4000, help="target number of rows")
    parser.add_argument("--no-balance", dest="balance", action="store_false",
                        help="keep every probed path instead of capping each class at samples / 4")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--extra-pairs", type=int, default=3, help="random host pairs probed per epoch")
    parser.add_argument("--probe-seconds", type=int, default=5, help="iperf3 probe length")
    parser.add_argument("--warmup", type=float, default=5.0, help="seconds after a new scenario before measuring")
    parser.add_argument("--path-hops", type=float, default=6.0, help="path length the class balance targets")
    parser.add_argument("--targeted", type=float, default=0.7,
                        help="share of epochs whose scenario targets the class furthest behind its quota")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--hosts-per-switch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mininet").propagate = False
    collect(args)


if __name__ == "__main__":
    main()
