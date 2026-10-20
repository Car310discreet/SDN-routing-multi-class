"""
telemetry.py - path features for the GEANT2 multi-class traffic classifier.

Active, end to end, on a path whose ICMP/TCP traffic is pinned by static entries:
    ping -c 50 -i 0.05 -W 5 src -> dst   rtt_avg_ms, rtt_min_ms, rtt_max_ms, rtt_mdev_ms (jitter),
                                         packet_loss_pct
    iperf3 TCP src -> dst, 5 s           throughput_mbps, retransmits,
                                         bw_utilization_pct = 100 * (1 - throughput / link capacity)
    path                                 hop_count
Passive, from the controller (Floodlight port counters read twice, `window_s` apart):
    path_max_link_util_pct, path_mean_link_util_pct  over both directions of the path's links

Passive features are measured once per decision (before any probe), so probe traffic never shows up
in them. After each iperf3 probe the queues are left to drain for `drain_s` seconds, so a probe does
not inflate the RTT of the next path.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geant2_topo  # noqa: E402,F401  (puts baseline/ on the path)
from telemetry_and_model import FloodlightClient  # noqa: E402

log = logging.getLogger(__name__)

ACTIVE_FEATURES = ["rtt_avg_ms", "rtt_min_ms", "rtt_max_ms", "rtt_mdev_ms", "packet_loss_pct",
                   "throughput_mbps", "bw_utilization_pct", "retransmits", "hop_count"]
PASSIVE_FEATURES = ["path_max_link_util_pct", "path_mean_link_util_pct"]
FEATURE_NAMES = ACTIVE_FEATURES + PASSIVE_FEATURES
PROBE_PORT = 5401


def parse_ping(output: str, deadline_s: float) -> dict:
    loss = re.search(r"([\d.]+)% packet loss", output)
    rtt = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", output)
    loss_pct = float(loss.group(1)) if loss else 100.0
    if rtt:
        rtt_min, rtt_avg, rtt_max, rtt_mdev = (float(x) for x in rtt.groups())
    else:                                  # nothing came back: as slow as the deadline allows
        rtt_min = rtt_avg = rtt_max = deadline_s * 1000.0
        rtt_mdev = 0.0
    return {"rtt_avg_ms": rtt_avg, "rtt_min_ms": rtt_min, "rtt_max_ms": rtt_max, "rtt_mdev_ms": rtt_mdev,
            "packet_loss_pct": loss_pct}


def parse_iperf(stdout: str, capacity_mbps: float) -> dict:
    try:
        end = json.loads(stdout)["end"]
    except (ValueError, KeyError, TypeError):
        end = {}
    received = end.get("sum_received") or {}
    sent = end.get("sum_sent") or {}
    throughput = float(received.get("bits_per_second", 0.0)) / 1e6
    return {"throughput_mbps": throughput,
            "bw_utilization_pct": min(max(100.0 * (1.0 - throughput / capacity_mbps), 0.0), 100.0),
            "retransmits": int(sent.get("retransmits", 0))}


class TelemetryClient(FloodlightClient):
    def port_tx_bytes(self) -> dict[tuple[str, int], int]:
        """(switch DPID, port) -> transmitted bytes, for every port of every connected switch."""
        counters = {}
        for dpid, reply in self._request("GET", "/wm/core/switch/all/port/json").items():
            if not isinstance(reply, dict):
                continue
            for part in reply.get("port_reply", []):
                for entry in part.get("port", []):
                    try:
                        counters[(dpid, int(entry["port_number"]))] = int(entry["transmit_bytes"])
                    except (KeyError, TypeError, ValueError):
                        continue             # the LOCAL port, or a malformed entry
        return counters


def link_utilization_from(spec, first: dict, second: dict, seconds: float) -> dict[tuple[str, str], float]:
    """Directed switch link (u, v) -> utilisation (fraction of capacity) from two counter snapshots."""
    capacity_bps = spec.link_bw_mbps * 1e6
    utilization = {}
    for a, b in spec.graph.edges:
        for u, v in ((a, b), (b, a)):
            key = (spec.switch_dpids[u], spec.port(u, v))
            sent = max(second.get(key, 0) - first.get(key, 0), 0) if key in first and key in second else 0
            utilization[(u, v)] = sent * 8 / max(seconds, 1e-3) / capacity_bps
    return utilization


def passive_features(link_util: dict[tuple[str, str], float], path: list[str]) -> dict:
    values = [link_util.get(hop, 0.0) for u, v in zip(path, path[1:]) for hop in ((u, v), (v, u))]
    return {"path_max_link_util_pct": 100.0 * max(values), "path_mean_link_util_pct": 100.0 * sum(values) / len(values)}


class PathTelemetry:
    def __init__(self, net, spec, api: TelemetryClient, ping_count: int = 50, ping_interval: float = 0.05,
                 ping_deadline_s: float = 5.0, probe_seconds: int = 5, window_s: float = 3.0, drain_s: float = 2.0,
                 port: int = PROBE_PORT):
        self.net, self.spec, self.api = net, spec, api
        self.ping_count, self.ping_interval, self.ping_deadline_s = ping_count, ping_interval, ping_deadline_s
        self.probe_seconds, self.window_s, self.drain_s, self.port = probe_seconds, window_s, drain_s, port

    def link_utilization(self) -> dict[tuple[str, str], float]:
        first, t0 = self.api.port_tx_bytes(), time.monotonic()
        time.sleep(self.window_s)
        second, t1 = self.api.port_tx_bytes(), time.monotonic()
        return link_utilization_from(self.spec, first, second, t1 - t0)

    def measure(self, src_host: str, dst_host: str, path: list[str], link_util: dict) -> dict:
        """The FEATURE_NAMES of `path` (ICMP/TCP between the hosts must already be pinned onto it)."""
        src, dst = self.net.get(src_host), self.net.get(dst_host)
        dst_ip = self.spec.hosts[dst_host].ip
        # exactly ping_count requests, then wait up to ping_deadline_s for late replies. (-w would keep
        # sending until ping_count replies arrive and count the requests still in flight as lost.)
        features = parse_ping(src.cmd(f"ping -q -c {self.ping_count} -i {self.ping_interval} "
                                      f"-W {self.ping_deadline_s:g} {dst_ip}"), self.ping_deadline_s)

        server = dst.popen(["iperf3", "-s", "-1", "-p", str(self.port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        client = src.popen(["iperf3", "-c", dst_ip, "-p", str(self.port), "-t", str(self.probe_seconds), "-J"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            stdout, _ = client.communicate(timeout=self.probe_seconds + 30)
        except subprocess.TimeoutExpired:
            client.kill()
            stdout = ""
        if server.poll() is None:
            server.terminate()
        features.update(parse_iperf(stdout, self.spec.link_bw_mbps))
        # the probe filled the path's queues (up to 1.2 s each): let them drain so the next path's
        # ping, which may share links, does not measure this probe
        time.sleep(self.drain_s)
        features["hop_count"] = len(path) - 1
        features.update(passive_features(link_util, path))
        return {name: float(features[name]) for name in FEATURE_NAMES}
