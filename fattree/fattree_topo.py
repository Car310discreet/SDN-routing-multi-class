"""
fattree_topo.py - a k=4 fat-tree data-centre network in Mininet + NetworkX, the second evaluation network.

Chosen to differ in character from GEANT2 (measured on the switch graphs):

    |                        | GEANT2      | fat-tree k=4                   |
    | switches / links       | 24 / 37     | 20 / 32                        |
    | degree                 | 2 - 8 (hub) | 4 core/aggregation, 2 edge     |
    | diameter               | 6           | 4                              |
    | algebraic connectivity | 0.31        | 0.76                           |
    | max betweenness        | 0.52 (s8)   | 0.14 (no hub)                  |
    | 5 hop-count candidates | 6,6,6,7,7   | 4,4,4,4,6 (four equal-cost)    |

Switches (k = 4: 4 core, 4 pods of 2 aggregation + 2 edge switches):
    core         s0 - s3      aggregation switch a of every pod -> core s(2a), s(2a+1)
    aggregation  s4 - s11     pod p: s(4+2p), s(5+2p)
    edge         s12 - s19    pod p: s(12+2p), s(13+2p); each linked to both aggregation switches of its pod
Links use the GEANT2 settings (10 Mbps, 10 ms, 1000-packet queue), so the topology is the only change.

Hosts: one per switch, numbered as in GEANT2 (h<i> = 10.0.0.<i+1> on s<i>). The traffic generator needs
a host on both ends of every link to pin each link's background flow; hosts on core and aggregation
switches are only background-traffic endpoints. Probes and the video use edge-switch hosts:
    Sender   h12 (10.0.0.13) on s12, pod 0   - streams the video
    Receiver h18 (10.0.0.19) on s18, pod 3   - receives and scores it

Offline summary (no root):   python fattree/fattree_topo.py
Interactive network (root):  sudo .venv/bin/python fattree/fattree_topo.py --cli   (Floodlight running)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import networkx as nx
from mininet.topo import Topo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "geant2"))
from geant2_topo import LINK_BW_MBPS, LINK_DELAY, QUEUE_SIZE, dpid, host_ip, host_name, start_geant2  # noqa: E402
from topology_builder import HostInfo, TopologySpec, stop_network  # noqa: E402  (baseline/, via geant2_topo)

logging.setLoggerClass(logging.Logger)
log = logging.getLogger(__name__)

K = 4
NUM_CORE, NUM_AGG, NUM_EDGE = (K // 2) ** 2, K * K // 2, K * K // 2
NUM_SWITCHES = NUM_CORE + NUM_AGG + NUM_EDGE
CORE = list(range(NUM_CORE))
AGGREGATION = list(range(NUM_CORE, NUM_CORE + NUM_AGG))
EDGE = list(range(NUM_CORE + NUM_AGG, NUM_SWITCHES))
SENDER, RECEIVER = "h12", "h18"
EDGE_SWITCHES = [f"s{i}" for i in EDGE]


def fat_tree_links(k: int = K) -> list[tuple[int, int]]:
    half, links = k // 2, []
    core, agg, edge = 0, half * half, half * half + k * half
    for pod in range(k):
        for a in range(half):
            for c in range(half):
                links.append((core + a * half + c, agg + pod * half + a))
        for e in range(half):
            for a in range(half):
                links.append((agg + pod * half + a, edge + pod * half + e))
    return links


FATTREE_LINKS = fat_tree_links()


class FatTreeTopo(Topo):
    def build(self, hosts_per_switch: int = 1, bw: float = LINK_BW_MBPS, delay: str = LINK_DELAY,
              queue: int = QUEUE_SIZE):
        for i in range(NUM_SWITCHES):
            self.addSwitch(f"s{i}", dpid=dpid(i), protocols="OpenFlow13")
        options = dict(bw=bw, delay=delay, max_queue_size=queue, use_htb=True)
        for a, b in FATTREE_LINKS:
            self.addLink(f"s{a}", f"s{b}", **options)
        for j in range(hosts_per_switch):
            for i in range(NUM_SWITCHES):
                host = self.addHost(host_name(i, j), ip=f"{host_ip(i, j)}/8")
                self.addLink(f"s{i}", host)          # access link: not shaped


def build_fattree(hosts_per_switch: int = 1, link_bw_mbps: float = LINK_BW_MBPS,
                  link_delay: str = LINK_DELAY) -> TopologySpec:
    if hosts_per_switch < 1:
        raise ValueError("hosts_per_switch must be >= 1")
    topo = FatTreeTopo(hosts_per_switch=hosts_per_switch, bw=link_bw_mbps, delay=link_delay)
    graph = nx.Graph()
    graph.add_nodes_from(f"s{i}" for i in range(NUM_SWITCHES))
    for a, b in FATTREE_LINKS:
        u, v = f"s{a}", f"s{b}"
        port_u, port_v = topo.port(u, v)
        graph.add_edge(u, v, ports={u: port_u, v: port_v})
    hosts = {}
    for j in range(hosts_per_switch):
        for i in range(NUM_SWITCHES):
            name = host_name(i, j)
            switch_port, _ = topo.port(f"s{i}", name)
            hosts[name] = HostInfo(name, host_ip(i, j), f"s{i}", switch_port)
    dpids = {f"s{i}": ":".join(dpid(i)[k:k + 2] for k in range(0, 16, 2)) for i in range(NUM_SWITCHES)}
    return TopologySpec(topo=topo, graph=graph, hosts=hosts, switch_dpids=dpids, link_bw_mbps=link_bw_mbps)


def start_fattree(spec: TopologySpec, api, ping_all: bool = True, **kwargs):
    """start_geant2() with the fat-tree Sender and Receiver as the pair that must always answer."""
    return start_geant2(spec, api, ping_all=ping_all, endpoints=(SENDER, RECEIVER), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", action="store_true", help="start the network and open the Mininet CLI (root)")
    parser.add_argument("--hosts-per-switch", type=int, default=1)
    parser.add_argument("--link-bw", type=float, default=LINK_BW_MBPS, help="Mbps per switch-switch link")
    parser.add_argument("--link-delay", default=LINK_DELAY)
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    spec = build_fattree(args.hosts_per_switch, args.link_bw, args.link_delay)
    sender, receiver = spec.hosts[SENDER], spec.hosts[RECEIVER]
    print(f"Fat-tree k={K}: {spec.graph.number_of_nodes()} switches, {spec.graph.number_of_edges()} links "
          f"({args.link_bw:g} Mbps, {args.link_delay}), {len(spec.hosts)} hosts; diameter "
          f"{nx.diameter(spec.graph)} hops")
    print(f"Sender   {SENDER} {sender.ip} on {sender.switch}\nReceiver {RECEIVER} {receiver.ip} on {receiver.switch} "
          f"({nx.shortest_path_length(spec.graph, sender.switch, receiver.switch)} hops apart)")
    if not args.cli:
        return
    if os.geteuid() != 0:
        raise SystemExit("Mininet needs root: run with sudo")
    from mininet.cli import CLI
    from telemetry_and_model import FloodlightClient
    api = FloodlightClient(args.controller_url)
    if not api.health():
        raise SystemExit(f"Floodlight is not reachable at {args.controller_url} (start ./start_floodlight.sh)")
    net = start_fattree(spec, api, ping_all=False)
    try:
        time.sleep(1)
        CLI(net)
    finally:
        stop_network(net)


if __name__ == "__main__":
    main()
