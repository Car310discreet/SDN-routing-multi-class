"""
geant2_topo.py - the GEANT2 evaluation network in Mininet + NetworkX.

GEANT2 (the pan-European research network) as used by RouteNet / DRL-GNN: 24 switches, 37 links
(edge list from knowledgedefinednetworking/DRL-GNN, gym_environments/envs/environment1.py).

Scaling knobs (all keep the GEANT2 switch graph):
  * hosts_per_switch  hosts on every switch (h<i> is the first host on s<i>, then h<i>n1, h<i>n2, ...);
                      extra hosts spread the background-traffic endpoints
  * link_bw_mbps      capacity of every switch-switch link (default 10 Mbps, as in the baseline)
  * link_delay        one-way delay of every switch-switch link (default 10 ms)
Host access links are not shaped, so a host never limits the background traffic it injects; the
switch-switch links are the only bottlenecks.

Sender and Receiver are at opposite edges of the network: s15 and s23 are 6 hops apart, the
diameter of GEANT2. Their 5 hop-count candidates do not all share links (one of them avoids the s8
hub), so path selection matters:
    Sender   h15 (10.0.0.16) on s15   - streams the video
    Receiver h23 (10.0.0.24) on s23   - receives and scores it

Offline summary (no root):   python geant2/geant2_topo.py
Interactive network (root):  sudo .venv/bin/python geant2/geant2_topo.py --cli   (Floodlight running)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import networkx as nx
from mininet.topo import Topo

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline"))
from topology_builder import HostInfo, TopologySpec, start_network, stop_network  # noqa: E402

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
LINK_BW_MBPS = 10
LINK_DELAY = "10ms"
QUEUE_SIZE = 1000
SENDER, RECEIVER = "h15", "h23"


def dpid(index: int) -> str:
    return f"{index + 1:016x}"


def host_name(switch_index: int, j: int = 0) -> str:
    return f"h{switch_index}" if j == 0 else f"h{switch_index}n{j}"


def host_ip(switch_index: int, j: int = 0) -> str:
    return f"10.{j}.0.{switch_index + 1}"


class Geant2Topo(Topo):
    def build(self, hosts_per_switch: int = 1, bw: float = LINK_BW_MBPS, delay: str = LINK_DELAY,
              queue: int = QUEUE_SIZE):
        for i in range(NUM_SWITCHES):
            self.addSwitch(f"s{i}", dpid=dpid(i), protocols="OpenFlow13")
        options = dict(bw=bw, delay=delay, max_queue_size=queue, use_htb=True)
        for a, b in GEANT2_LINKS:
            self.addLink(f"s{a}", f"s{b}", **options)
        for j in range(hosts_per_switch):
            for i in range(NUM_SWITCHES):
                host = self.addHost(host_name(i, j), ip=f"{host_ip(i, j)}/8")
                self.addLink(f"s{i}", host)          # access link: not shaped


def build_geant2(hosts_per_switch: int = 1, link_bw_mbps: float = LINK_BW_MBPS,
                 link_delay: str = LINK_DELAY) -> TopologySpec:
    if hosts_per_switch < 1:
        raise ValueError("hosts_per_switch must be >= 1")
    topo = Geant2Topo(hosts_per_switch=hosts_per_switch, bw=link_bw_mbps, delay=link_delay)
    graph = nx.Graph()
    graph.add_nodes_from(f"s{i}" for i in range(NUM_SWITCHES))
    for a, b in GEANT2_LINKS:
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


def hosts_on(spec: TopologySpec, switch: str) -> list[str]:
    """Hosts attached to `switch`, first host first."""
    return sorted((h for h, info in spec.hosts.items() if info.switch == switch), key=lambda h: (len(h), h))


IPV6_DEFAULT = "/proc/sys/net/ipv6/conf/default/disable_ipv6"


def start_without_ipv6(spec: TopologySpec):
    """
    start_network() with IPv6 off on every switch and host interface, and static ARP entries.

    With IPv6 on, the link-local traffic of the switch ports (in the root namespace) reaches Open vSwitch
    with a packet mark. Open vSwitch then reports a pkt_mark field in PacketIn messages, which
    Floodlight's OpenFlow library cannot parse (OFParseError ... OFOxmVer13: 82436). Floodlight drops
    the switch connection and wipes its flow table; with many switches its routing state can stay
    stale (whole switches unreachable). Measured on a 4-switch network: 22 parse errors in 45 s with
    IPv6, none without.
    """
    with open(IPV6_DEFAULT) as handle:
        previous = handle.read().strip()
    with open(IPV6_DEFAULT, "w") as handle:      # only interfaces created while this is set
        handle.write("1")
    try:
        net = start_network(spec)
    finally:
        with open(IPV6_DEFAULT, "w") as handle:
            handle.write(previous)
    for host in net.hosts:
        host.cmd("sysctl -qw net.ipv6.conf.all.disable_ipv6=1")
    # Static ARP: static entries pin only IPv4 onto a path, so ARP would otherwise follow Floodlight's
    # own route, and on an overloaded link a lost ARP exchange looks like 100 % ping loss on the probed path
    net.staticArp()
    return net


def start_geant2(spec: TopologySpec, api, ping_all: bool = True, settle_s: float = 30.0,
                 max_loss_pct: float = 1.0, endpoints: tuple[str, str] = (SENDER, RECEIVER)):
    """
    Start GEANT2 on Floodlight, wait for every switch, clear old static entries and check connectivity
    (needs root).

    pingAll sends one ping per host pair with a 1 s timeout, so a few pairs can time out while
    Floodlight sets up reactive flows. Up to `max_loss_pct` of isolated misses is accepted. A broken
    controller loses whole hosts: 46 of the 552 pairs per unreachable host, more than 8 %. The
    Sender <-> Receiver pair (`endpoints`, also used for the fat-tree) must always answer.
    """
    from telemetry_and_model import ControllerError, wait_for_switches
    net = start_without_ipv6(spec)
    try:
        wait_for_switches(api, set(spec.switch_dpids.values()))
        api.clear_flows()            # static entries of earlier runs reappear on reused DPIDs
        api.set_routing_metric("hopcount")
        if ping_all:
            log.info("Testing connectivity (pingAll) so Floodlight learns every host")
            endpoints = [net.get(host) for host in endpoints]
            for attempt in range(2):
                loss = net.pingAll()
                pair_loss = net.ping(endpoints, timeout="2")
                if loss <= max_loss_pct and pair_loss == 0:
                    break
                if attempt == 0:
                    log.warning("pingAll lost %.1f %% (Sender <-> Receiver %.0f %%); waiting %.0f s for the "
                                "topology to settle", loss, pair_loss, settle_s)
                    time.sleep(settle_s)
            else:
                raise ControllerError(f"Hosts cannot reach each other through Floodlight ({loss:.1f} % ping loss, "
                                      f"Sender <-> Receiver {pair_loss:.0f} %)")
    except BaseException:
        stop_network(net)
        raise
    return net


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cli", action="store_true", help="start the network and open the Mininet CLI (root)")
    parser.add_argument("--hosts-per-switch", type=int, default=1)
    parser.add_argument("--link-bw", type=float, default=LINK_BW_MBPS, help="Mbps per switch-switch link")
    parser.add_argument("--link-delay", default=LINK_DELAY)
    parser.add_argument("--controller-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    spec = build_geant2(args.hosts_per_switch, args.link_bw, args.link_delay)
    sender, receiver = spec.hosts[SENDER], spec.hosts[RECEIVER]
    print(f"GEANT2: {spec.graph.number_of_nodes()} switches, {spec.graph.number_of_edges()} links "
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
    net = start_geant2(spec, api, ping_all=False)
    try:
        time.sleep(1)
        CLI(net)
    finally:
        stop_network(net)


if __name__ == "__main__":
    main()
