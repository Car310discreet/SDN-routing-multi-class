"""
traffic_gen.py - artificial background traffic on GEANT2 in four traffic classes.

Every switch-switch link carries one iperf3 UDP flow in each direction between hosts on its two
ends, pinned onto that link with static entries (so the offered load of each link is known exactly).
The offered load of a link, relative to its capacity, falls in one of four classes:

    0  Extremely Low   [0.00, 0.25)
    1  Low             [0.25, 0.50)
    2  High            [0.50, 0.80)
    3  Extremely High  [0.80, 1.20]    (above 1.0 the link queue fills and drops packets)

A Scenario assigns a load to every link (the same in both directions). Every epoch a new scenario
is drawn, so link utilisation fluctuates over time. A path's class is the class of its busiest link
(its bottleneck). Drawing each link's class uniformly would make almost every multi-hop path
"Extremely High", so link classes are drawn with weights that make path classes balanced for paths
of `path_hops` links: P(link class <= c) = ((c + 1) / 4) ** (1 / path_hops).

A targeted scenario (`target_class=c`) draws link classes only up to c, with class c on a
`target_share` of the links, so most multi-hop paths have a class-c bottleneck. The dataset collector
uses it to fill the classes that lag behind. Labels are still the true bottleneck class of each path.

Offline preview of the class balance (no root):
    python geant2/traffic_gen.py --preview
Fluctuating traffic on a live network (root, Floodlight running), new scenario every 60 s:
    sudo .venv/bin/python geant2/traffic_gen.py --epoch 60
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geant2_topo import RECEIVER, SENDER, build_geant2, hosts_on  # noqa: E402
from path_discovery import yen_hop_count  # noqa: E402  (baseline/, put on the path by geant2_topo)

log = logging.getLogger(__name__)

CLASS_NAMES = ["Extremely Low", "Low", "High", "Extremely High"]
CLASS_BANDS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.80), (0.80, 1.20)]
PORT_BASE = 7000
PIN_PRIORITY = 32768
MIN_RATE_MBPS = 0.05             # below this no flow is started
DAY_S = 86400


def class_of(utilization: float) -> int:
    for index, (_, high) in enumerate(CLASS_BANDS[:-1]):
        if utilization < high:
            return index
    return len(CLASS_BANDS) - 1


def link_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if int(u[1:]) < int(v[1:]) else (v, u)


def link_class_weights(path_hops: float) -> np.ndarray:
    """P(link class = c) such that the max class over `path_hops` links is uniform over the 4 classes."""
    cumulative = np.array([((c + 1) / len(CLASS_NAMES)) ** (1.0 / path_hops) for c in range(len(CLASS_NAMES))])
    return np.diff(np.concatenate([[0.0], cumulative]))


@dataclass
class Scenario:
    utilization: dict[tuple[str, str], float]      # link_key -> offered load / capacity (each direction)
    seed: int | None = None

    def link(self, u: str, v: str) -> float:
        return self.utilization[link_key(u, v)]

    def path_utilization(self, path: list[str]) -> float:
        return max(self.link(u, v) for u, v in zip(path, path[1:]))

    def path_class(self, path: list[str]) -> int:
        return class_of(self.path_utilization(path))

    def link_classes(self) -> Counter:
        return Counter(class_of(u) for u in self.utilization.values())

    def to_dict(self) -> dict:
        return {"seed": self.seed, "utilization": {f"{u}-{v}": round(x, 4) for (u, v), x in self.utilization.items()}}


def targeted_weights(target_class: int, path_hops: float, target_share: float = 0.3) -> np.ndarray:
    """Link-class weights with no class above `target_class` and `target_share` of links in it."""
    weights = link_class_weights(path_hops)
    weights[target_class + 1:] = 0.0
    if target_class == 0:
        return weights / weights.sum()
    lower = weights[:target_class] / weights[:target_class].sum() * (1.0 - target_share)
    weights[:target_class], weights[target_class] = lower, target_share
    return weights


def sample_scenario(graph, seed: int | None = None, path_hops: float = 6.0,
                    target_class: int | None = None) -> Scenario:
    rng = np.random.default_rng(seed)
    weights = link_class_weights(path_hops) if target_class is None else targeted_weights(target_class, path_hops)
    utilization = {}
    for u, v in sorted(link_key(a, b) for a, b in graph.edges):
        low, high = CLASS_BANDS[rng.choice(len(CLASS_BANDS), p=weights)]
        utilization[(u, v)] = float(rng.uniform(low, high))
    return Scenario(utilization, seed)


@dataclass
class TrafficGenerator:
    """Pinned per-link iperf3 UDP flows whose rates follow the current Scenario (needs Mininet)."""
    net: object
    spec: object
    installer: object                          # telemetry_and_model.FlowInstaller
    port_base: int = PORT_BASE
    flows: list = field(default_factory=list)  # (link_key, sender host, receiver host, port)
    servers: list = field(default_factory=list)
    clients: list = field(default_factory=list)
    flow_names: list = field(default_factory=list)
    scenario: Scenario | None = None

    def _endpoints(self, link_index: int, u: str, v: str) -> tuple[str, str]:
        """Spread links over the hosts of each switch when there are several per switch."""
        on_u, on_v = hosts_on(self.spec, u), hosts_on(self.spec, v)
        return on_u[link_index % len(on_u)], on_v[link_index % len(on_v)]

    def start(self) -> None:
        """Pin every link's endpoint pair onto that link and start the iperf3 servers."""
        for index, (u, v) in enumerate(sorted(link_key(a, b) for a, b in self.spec.graph.edges)):
            a, b = self._endpoints(index, u, v)
            self.flow_names += self.installer.install_path(self.spec.hosts[a].ip, self.spec.hosts[b].ip, [u, v],
                                                           prefix=f"bg_{u}_{v}", priority=PIN_PRIORITY)
            for direction, (src, dst) in enumerate(((a, b), (b, a))):
                port = self.port_base + 2 * index + direction
                self.flows.append(((u, v), src, dst, port))
                self.servers.append(self.net.get(dst).popen(["iperf3", "-s", "-p", str(port)],
                                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(1)
        log.info("Pinned %d background flows on %d links", len(self.flows), self.spec.graph.number_of_edges())

    def _stop_clients(self) -> None:
        for proc in self.clients:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.clients:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.clients = []

    def apply(self, scenario: Scenario) -> None:
        """Restart the clients at the rates of `scenario` (both directions of a link get the same load)."""
        self._stop_clients()
        if self.scenario is not None:
            time.sleep(1)                  # let the servers reset after the previous test
        for key, src, dst, port in self.flows:
            rate = scenario.utilization[key] * self.spec.link_bw_mbps
            if rate < MIN_RATE_MBPS:
                continue
            self.clients.append(self.net.get(src).popen(
                ["iperf3", "-c", self.spec.hosts[dst].ip, "-p", str(port), "-u", "-b", f"{rate:.3f}M",
                 "-t", str(DAY_S)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.scenario = scenario
        counts = scenario.link_classes()
        log.info("Scenario %s: links per class %s", scenario.seed,
                 ", ".join(f"{CLASS_NAMES[c]} {counts.get(c, 0)}" for c in range(len(CLASS_NAMES))))

    def stop(self) -> None:
        self._stop_clients()
        for proc in self.servers:
            if proc.poll() is None:
                proc.terminate()
        self.installer.remove(self.flow_names)
        self.servers, self.flows, self.flow_names = [], [], []


def preview(samples: int, path_hops: float) -> None:
    """Class balance of the Sender -> Receiver candidates over many sampled scenarios (offline)."""
    spec = build_geant2()
    candidates = yen_hop_count(spec.graph, spec.hosts[SENDER].switch, spec.hosts[RECEIVER].switch, 5)
    link_counts, path_counts = Counter(), Counter()
    for seed in range(samples):
        scenario = sample_scenario(spec.graph, seed, path_hops)
        link_counts.update(scenario.link_classes())
        path_counts.update(scenario.path_class(p) for p in candidates)
    print(f"Link-class weights for {path_hops:g}-hop paths: "
          + ", ".join(f"{n} {w:.3f}" for n, w in zip(CLASS_NAMES, link_class_weights(path_hops))))
    for title, counts in (("links", link_counts), ("candidate paths", path_counts)):
        total = sum(counts.values())
        print(f"{title:>16}: " + ", ".join(f"{CLASS_NAMES[c]} {counts[c] / total:.1%}" for c in range(4)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true", help="print the class balance only (no network)")
    parser.add_argument("--epoch", type=float, default=60, help="seconds between scenarios")
    parser.add_argument("--path-hops", type=float, default=6.0, help="path length the class balance targets")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hosts-per-switch", type=int, default=1)
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mininet").propagate = False
    if args.preview:
        preview(2000, args.path_hops)
        return
    if os.geteuid() != 0:
        raise SystemExit("Mininet needs root: run with sudo")

    from geant2_topo import start_geant2
    from telemetry_and_model import FloodlightClient, FlowInstaller
    from topology_builder import stop_network
    api = FloodlightClient(args.controller_url)
    if not api.health():
        raise SystemExit(f"Floodlight is not reachable at {args.controller_url} (start ./start_floodlight.sh)")
    spec = build_geant2(args.hosts_per_switch)
    rng = random.Random(args.seed)
    net = generator = None
    try:
        net = start_geant2(spec, api, ping_all=False)
        generator = TrafficGenerator(net, spec, FlowInstaller(api, spec))
        generator.start()
        while True:
            generator.apply(sample_scenario(spec.graph, rng.randrange(2 ** 31), args.path_hops))
            time.sleep(args.epoch)
    except KeyboardInterrupt:
        log.info("Stopping")
    finally:
        if generator is not None:
            generator.stop()
        stop_network(net)


if __name__ == "__main__":
    main()
