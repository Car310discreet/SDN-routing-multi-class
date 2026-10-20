"""
topology_builder.py - Module 1: the GEANT2 network and its 4-class background traffic.

Topology (Mininet + NetworkX)
  GEANT2 as used by RouteNet / DRL-GNN: 24 switches, 37 links (edge list from
  knowledgedefinednetworking/DRL-GNN). Switch-switch links default to 10 Mbit/s, 10 ms, a
  1,000-packet queue: the conditions the GEANT2 multi-class dataset was collected under, so Stage 2
  features stay in the model's training distribution. Hosts: one per switch (h<i> on s<i>, 10.0.0.<i+1>);
  access links are not shaped.
      Sender   h15 (s15)  - video source
      Receiver h23 (s23)  - video sink, 6 hops away (the network diameter)
  `--delay-range LO HI` draws per-link propagation delays instead (seeded), which makes the latency
  metric discriminative but moves RTT features away from the training data.

Startup (`start_geant2`)
  IPv6 is disabled on every interface while the network is built: with it, Open vSwitch reports a
  pkt_mark field in PacketIns that Floodlight cannot parse, so Floodlight drops switches and can keep
  stale routes. Hosts get static ARP. pingAll must succeed (a few isolated timeouts allowed) and the
  Sender <-> Receiver pair must answer.

Traffic (`TrafficGenerator`, `Scenario`)
  Every link carries one iperf3 UDP flow per direction between the hosts at its ends, pinned onto that
  link with static entries, at a rate set by the link's class:
      0 Extremely Low [0.00, 0.25)   1 Low [0.25, 0.50)   2 High [0.50, 0.80)   3 Extremely High [0.80, 1.20]
  (fraction of capacity). A path's ground-truth class is the class of its most loaded link. Link classes
  are drawn so that 6-hop path classes are balanced: P(link class <= c) = ((c + 1) / 4) ** (1 / 6).

    python advanced/topology_builder.py                 # offline summary
    python advanced/topology_builder.py --preview       # class balance of sampled scenarios
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from mininet.topo import Topo

# Importing Mininet makes every later logger a level-25 MininetLogger, which hides INFO messages
logging.setLoggerClass(logging.Logger)
log = logging.getLogger(__name__)

NUM_SWITCHES = 24
GEANT2_LINKS = [
    (0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
    (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
    (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12, 19), (12, 21),
    (14, 15), (15, 16), (16, 17), (17, 18), (18, 21), (19, 23), (21, 22), (22, 23),
]
LINK_BW_MBPS = 10.0
LINK_DELAY_MS = 10.0
QUEUE_SIZE = 1000
SENDER, RECEIVER = "h15", "h23"

CLASS_NAMES = ["Extremely Low", "Low", "High", "Extremely High"]
CLASS_BANDS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.80), (0.80, 1.20)]


# ----------------------------------------------------------------------------------------
# Topology
# ----------------------------------------------------------------------------------------
@dataclass
class HostInfo:
    name: str
    ip: str
    switch: str
    switch_port: int


@dataclass
class TopologySpec:
    topo: Topo
    graph: nx.Graph                      # switch graph; edges: ports, delay_ms, capacity_mbps
    hosts: dict[str, HostInfo]
    switch_dpids: dict[str, str]         # switch -> Floodlight DPID (colon form)
    link_bw_mbps: float = LINK_BW_MBPS

    @property
    def dpid_to_switch(self) -> dict[str, str]:
        return {d: s for s, d in self.switch_dpids.items()}

    def port(self, u: str, v: str) -> int:
        """Port on switch u of the link to switch v."""
        return self.graph.edges[u, v]["ports"][u]

    def host_by_ip(self, ip: str) -> HostInfo:
        for info in self.hosts.values():
            if info.ip == ip:
                return info
        raise KeyError(f"No host with IP {ip}")

    def access_interface(self, host: str) -> str:
        """Root-namespace interface of the switch port facing `host` (e.g. s23-eth3)."""
        info = self.hosts[host]
        return f"{info.switch}-eth{info.switch_port}"

    def hosts_on(self, switch: str) -> list[str]:
        return sorted((h for h, i in self.hosts.items() if i.switch == switch), key=lambda h: (len(h), h))


def dpid(index: int) -> str:
    return f"{index + 1:016x}"


class Geant2Topo(Topo):
    def build(self, delays: dict, bw: float = LINK_BW_MBPS, queue: int = QUEUE_SIZE):
        for i in range(NUM_SWITCHES):
            self.addSwitch(f"s{i}", dpid=dpid(i), protocols="OpenFlow13")
        for a, b in GEANT2_LINKS:
            self.addLink(f"s{a}", f"s{b}", bw=bw, delay=f"{delays[(a, b)]}ms", max_queue_size=queue, use_htb=True)
        for i in range(NUM_SWITCHES):
            host = self.addHost(f"h{i}", ip=f"10.0.0.{i + 1}/8")
            self.addLink(f"s{i}", host)          # access link: not shaped


def build_geant2(link_bw_mbps: float = LINK_BW_MBPS, delay_range_ms: tuple[float, float] | None = None,
                 seed: int = 1) -> TopologySpec:
    if delay_range_ms is None:
        delays = {link: LINK_DELAY_MS for link in GEANT2_LINKS}
    else:
        rng = random.Random(seed)
        delays = {link: round(rng.uniform(*delay_range_ms), 1) for link in GEANT2_LINKS}
    topo = Geant2Topo(delays=delays, bw=link_bw_mbps)
    graph = nx.Graph()
    graph.add_nodes_from(f"s{i}" for i in range(NUM_SWITCHES))
    for a, b in GEANT2_LINKS:
        u, v = f"s{a}", f"s{b}"
        port_u, port_v = topo.port(u, v)
        graph.add_edge(u, v, ports={u: port_u, v: port_v}, delay_ms=delays[(a, b)], capacity_mbps=link_bw_mbps)
    hosts = {}
    for i in range(NUM_SWITCHES):
        switch_port, _ = topo.port(f"s{i}", f"h{i}")
        hosts[f"h{i}"] = HostInfo(f"h{i}", f"10.0.0.{i + 1}", f"s{i}", switch_port)
    dpids = {f"s{i}": ":".join(dpid(i)[k:k + 2] for k in range(0, 16, 2)) for i in range(NUM_SWITCHES)}
    return TopologySpec(topo, graph, hosts, dpids, link_bw_mbps)


# ----------------------------------------------------------------------------------------
# Network lifecycle (root)
# ----------------------------------------------------------------------------------------
IPV6_DEFAULT = "/proc/sys/net/ipv6/conf/default/disable_ipv6"


def start_geant2(spec: TopologySpec, api, ping_all: bool = True, settle_s: float = 30.0,
                 max_loss_pct: float = 1.0, controller_ip="127.0.0.1", controller_port=6653):
    """Start GEANT2 on Floodlight with IPv6 off and static ARP; check connectivity. Needs root."""
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import OVSKernelSwitch, RemoteController
    from telemetry_and_model import ControllerError, wait_for_switches

    with open(IPV6_DEFAULT) as handle:
        previous = handle.read().strip()
    with open(IPV6_DEFAULT, "w") as handle:        # only interfaces created while this is set
        handle.write("1")
    net = None
    try:
        net = Mininet(topo=spec.topo, switch=OVSKernelSwitch, link=TCLink, controller=None,
                      autoSetMacs=True, build=False, waitConnected=True)
        net.addController("c1", controller=RemoteController, ip=controller_ip, port=controller_port,
                          protocols="OpenFlow13")
        net.build()
        net.start()
    except BaseException:
        if net is not None:
            net.stop()
        raise
    finally:
        with open(IPV6_DEFAULT, "w") as handle:
            handle.write(previous)
    try:
        for host in net.hosts:
            host.cmd("sysctl -qw net.ipv6.conf.all.disable_ipv6=1")
        net.staticArp()
        wait_for_switches(api, set(spec.switch_dpids.values()))
        api.clear_flows()            # static entries of earlier runs reappear on reused DPIDs
        api.set_routing_metric("hopcount")
        if ping_all:
            endpoints = [net.get(SENDER), net.get(RECEIVER)]
            for attempt in range(2):
                loss = net.pingAll()
                pair_loss = net.ping(endpoints, timeout="2")
                if loss <= max_loss_pct and pair_loss == 0:
                    break
                if attempt == 0:
                    log.warning("pingAll lost %.1f %% (Sender <-> Receiver %.0f %%); waiting %.0f s", loss,
                                pair_loss, settle_s)
                    time.sleep(settle_s)
            else:
                raise ControllerError(f"Hosts cannot reach each other through Floodlight ({loss:.1f} % loss)")
    except BaseException:
        stop_network(net)
        raise
    return net


def stop_network(net) -> None:
    if net is None:
        return
    for host in net.hosts:
        if host.waiting:
            try:
                host.sendInt()
                host.waitOutput()
            except Exception as exc:
                log.warning("Could not interrupt %s: %s", host.name, exc)
    net.stop()


def chown_to_invoking_user(path: str) -> None:
    """Outputs are written as root under sudo; hand them back to the user who ran sudo."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid is None or gid is None or not os.path.exists(path):
        return
    for root, dirs, files in os.walk(path):
        for name in [root] + [os.path.join(root, n) for n in dirs + files]:
            try:
                os.chown(name, int(uid), int(gid))
            except OSError:
                pass


# ----------------------------------------------------------------------------------------
# 4-class traffic
# ----------------------------------------------------------------------------------------
def class_of(utilization: float) -> int:
    for index, (_, high) in enumerate(CLASS_BANDS[:-1]):
        if utilization < high:
            return index
    return len(CLASS_BANDS) - 1


def link_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if int(u[1:]) < int(v[1:]) else (v, u)


def link_class_weights(path_hops: float = 6.0) -> np.ndarray:
    """P(link class = c) such that the max class over `path_hops` links is uniform over the classes."""
    cumulative = np.array([((c + 1) / len(CLASS_NAMES)) ** (1.0 / path_hops) for c in range(len(CLASS_NAMES))])
    return np.diff(np.concatenate([[0.0], cumulative]))


@dataclass
class Scenario:
    utilization: dict[tuple[str, str], float]      # link_key -> offered load / capacity, each direction
    seed: int | None = None

    def link(self, u: str, v: str) -> float:
        return self.utilization[link_key(u, v)]

    def path_utilization(self, path: list[str]) -> float:
        return max(self.link(u, v) for u, v in zip(path, path[1:]))

    def path_class(self, path: list[str]) -> int:
        return class_of(self.path_utilization(path))

    def congest(self, path: list[str], utilization: float = 1.0) -> None:
        """Put every link of `path` in Extremely High (the AIRSDN scenario: congested default path)."""
        for u, v in zip(path, path[1:]):
            self.utilization[link_key(u, v)] = utilization

    def to_dict(self) -> dict:
        return {"seed": self.seed, "utilization": {f"{u}-{v}": round(x, 4) for (u, v), x in self.utilization.items()}}


def sample_scenario(graph: nx.Graph, seed: int | None = None, path_hops: float = 6.0) -> Scenario:
    rng = np.random.default_rng(seed)
    weights = link_class_weights(path_hops)
    utilization = {}
    for u, v in sorted(link_key(a, b) for a, b in graph.edges):
        low, high = CLASS_BANDS[rng.choice(len(CLASS_BANDS), p=weights)]
        utilization[(u, v)] = float(rng.uniform(low, high))
    return Scenario(utilization, seed)


@dataclass
class TrafficGenerator:
    """Pinned per-link iperf3 UDP flows whose rates follow the current Scenario (needs Mininet)."""
    net: object
    spec: TopologySpec
    installer: object                          # telemetry_and_model.FlowInstaller
    port_base: int = 7000
    flows: list = field(default_factory=list)  # (link_key, sender host, receiver host, port)
    servers: list = field(default_factory=list)
    clients: list = field(default_factory=list)
    flow_names: list = field(default_factory=list)
    scenario: Scenario | None = None

    PIN_PRIORITY = 32768
    MIN_RATE_MBPS = 0.05
    DURATION_S = 86400

    def start(self) -> None:
        for index, (u, v) in enumerate(sorted(link_key(a, b) for a, b in self.spec.graph.edges)):
            a, b = self.spec.hosts_on(u)[0], self.spec.hosts_on(v)[0]
            self.flow_names += self.installer.install_path(self.spec.hosts[a].ip, self.spec.hosts[b].ip, [u, v],
                                                           prefix=f"bg_{u}_{v}", priority=self.PIN_PRIORITY)
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
            if rate < self.MIN_RATE_MBPS:
                continue
            self.clients.append(self.net.get(src).popen(
                ["iperf3", "-c", self.spec.hosts[dst].ip, "-p", str(port), "-u", "-b", f"{rate:.3f}M",
                 "-t", str(self.DURATION_S)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self.scenario = scenario
        counts = Counter(class_of(u) for u in scenario.utilization.values())
        log.info("Scenario %s: links per class %s", scenario.seed,
                 ", ".join(f"{CLASS_NAMES[c]} {counts.get(c, 0)}" for c in range(len(CLASS_NAMES))))

    def stop(self) -> None:
        self._stop_clients()
        for proc in self.servers:
            if proc.poll() is None:
                proc.terminate()
        self.installer.remove(self.flow_names)
        self.servers, self.flows, self.flow_names = [], [], []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true", help="class balance of 2,000 sampled scenarios")
    args = parser.parse_args()
    spec = build_geant2()
    s, r = spec.hosts[SENDER], spec.hosts[RECEIVER]
    print(f"GEANT2: {spec.graph.number_of_nodes()} switches, {spec.graph.number_of_edges()} links, "
          f"diameter {nx.diameter(spec.graph)}; Sender {SENDER} on {s.switch}, Receiver {RECEIVER} on {r.switch} "
          f"({nx.shortest_path_length(spec.graph, s.switch, r.switch)} hops)")
    if args.preview:
        from path_discovery import discover_paths
        candidates = [p.switches for p in discover_paths(spec.graph, s.switch, r.switch, "hop", 5).paths]
        counts = Counter(sample_scenario(spec.graph, seed).path_class(p) for seed in range(2000) for p in candidates)
        total = sum(counts.values())
        print("Hop-count candidates per class: " + ", ".join(f"{CLASS_NAMES[c]} {counts[c] / total:.1%}"
                                                              for c in range(4)))


if __name__ == "__main__":
    main()
