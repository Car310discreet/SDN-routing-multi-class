"""Offline tests for the fat-tree topology (run: .venv/bin/python -m pytest fattree/tests).

No Mininet network or Floodlight: only the topology, its candidates and the dataset collector's probe choice.
"""
import itertools
import os
import random
import sys
from collections import Counter

import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "geant2"))
import collect_dataset  # noqa: E402
from fattree_topo import EDGE_SWITCHES, RECEIVER, SENDER, build_fattree  # noqa: E402  (put on the path by collect_dataset)
from path_discovery import yen_hop_count  # noqa: E402
from traffic_gen import sample_scenario  # noqa: E402

SPEC = build_fattree()
CANDIDATES = yen_hop_count(SPEC.graph, SPEC.hosts[SENDER].switch, SPEC.hosts[RECEIVER].switch, 5)


def test_fattree_shape_and_endpoints():
    graph = SPEC.graph
    assert graph.number_of_nodes() == 20 and graph.number_of_edges() == 32 and len(SPEC.hosts) == 20
    degree = dict(graph.degree())
    assert all(degree[f"s{i}"] == 4 for i in range(12)) and all(degree[s] == 2 for s in EDGE_SWITCHES)
    src, dst = SPEC.hosts[SENDER].switch, SPEC.hosts[RECEIVER].switch
    assert src in EDGE_SWITCHES and dst in EDGE_SWITCHES
    assert nx.shortest_path_length(graph, src, dst) == nx.diameter(graph) == 4
    for switch in graph.nodes:                   # every port on a switch is used once
        ports = [SPEC.port(switch, n) for n in graph.neighbors(switch)]
        ports += [h.switch_port for h in SPEC.hosts.values() if h.switch == switch]
        assert len(ports) == len(set(ports))
    assert len(set(SPEC.switch_dpids.values())) == 20


def test_four_equal_cost_candidates():
    assert [len(p) - 1 for p in CANDIDATES] == [4, 4, 4, 4, 6]
    reference = [len(p) - 1 for p in itertools.islice(
        nx.shortest_simple_paths(SPEC.graph, CANDIDATES[0][0], CANDIDATES[0][-1]), 5)]
    assert [len(p) - 1 for p in CANDIDATES] == reference
    assert len({p[2] for p in CANDIDATES[:4]}) == 4             # each through a different core switch


def test_extra_probes_are_between_edge_switches():
    topology = collect_dataset.TOPOLOGIES["fattree"]
    rng = random.Random(0)
    for _ in range(50):
        probes = collect_dataset.epoch_paths(SPEC, rng, CANDIDATES, 3, SENDER, RECEIVER, topology["probe_switches"])
        assert len(probes) == 8 and all((s, d) == (SENDER, RECEIVER) for s, d, _ in probes[:5])
        for src, dst, path in probes[5:]:
            assert SPEC.hosts[src].switch == path[0] in EDGE_SWITCHES
            assert SPEC.hosts[dst].switch == path[-1] in EDGE_SWITCHES


def test_candidate_classes_are_balanced():
    path_hops = collect_dataset.TOPOLOGIES["fattree"]["path_hops"]
    counts = Counter(sample_scenario(SPEC.graph, seed, path_hops).path_class(p)
                     for seed in range(1000) for p in CANDIDATES)
    assert all(0.18 < counts[c] / sum(counts.values()) < 0.32 for c in range(4))
