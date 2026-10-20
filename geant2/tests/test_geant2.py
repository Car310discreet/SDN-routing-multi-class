"""Offline tests for the GEANT2 modules (run: .venv/bin/python -m pytest geant2/tests).

No Mininet or Floodlight: datasets and run results are synthetic and live in pytest's tmp_path.
"""
import itertools
import json
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import evaluate  # noqa: E402
import train_models  # noqa: E402
from controller_routing import candidate_paths, ml_routing  # noqa: E402
from geant2_topo import RECEIVER, SENDER, build_geant2  # noqa: E402
from ml_classifier import CongestionClassifier  # noqa: E402
from path_discovery import yen_hop_count  # noqa: E402
from telemetry import FEATURE_NAMES, link_utilization_from, parse_iperf, parse_ping, passive_features  # noqa: E402
from traffic_gen import CLASS_BANDS, class_of, link_class_weights, sample_scenario  # noqa: E402

SPEC = build_geant2()


def test_geant2_shape_and_endpoints():
    assert SPEC.graph.number_of_nodes() == 24 and SPEC.graph.number_of_edges() == 37
    assert nx.is_connected(SPEC.graph) and len(SPEC.hosts) == 24
    src, dst = SPEC.hosts[SENDER].switch, SPEC.hosts[RECEIVER].switch
    assert nx.shortest_path_length(SPEC.graph, src, dst) == nx.diameter(SPEC.graph) == 6
    for switch in SPEC.graph.nodes:              # every port on a switch is used once
        ports = [SPEC.port(switch, n) for n in SPEC.graph.neighbors(switch)]
        ports += [h.switch_port for h in SPEC.hosts.values() if h.switch == switch]
        assert len(ports) == len(set(ports))
    assert len(set(SPEC.switch_dpids.values())) == 24


def test_scaled_topology_has_more_hosts():
    spec = build_geant2(hosts_per_switch=3, link_bw_mbps=20)
    assert len(spec.hosts) == 72 and spec.link_bw_mbps == 20
    assert len({h.ip for h in spec.hosts.values()}) == 72


def test_candidates_are_top5_by_hop_count():
    paths = candidate_paths(SPEC)
    assert len(paths) == 5 and len({tuple(p) for p in paths}) == 5
    reference = [len(p) - 1 for p in itertools.islice(
        nx.shortest_simple_paths(SPEC.graph, paths[0][0], paths[0][-1]), 5)]
    assert [len(p) - 1 for p in paths] == reference
    assert any("s8" not in p for p in paths)      # one candidate avoids the hub


def test_traffic_classes_and_balance():
    assert [class_of(x) for x in (0.0, 0.24, 0.25, 0.6, 0.8, 1.19)] == [0, 0, 1, 2, 3, 3]
    assert np.isclose(link_class_weights(6).sum(), 1.0)
    scenario = sample_scenario(SPEC.graph, seed=3)
    for u, v in SPEC.graph.edges:
        low, high = CLASS_BANDS[class_of(scenario.link(u, v))]
        assert low <= scenario.link(v, u) < high or scenario.link(u, v) >= CLASS_BANDS[-1][0]
    candidates = candidate_paths(SPEC)
    counts = Counter(sample_scenario(SPEC.graph, s).path_class(p) for s in range(400) for p in candidates)
    assert all(0.15 < counts[c] / sum(counts.values()) < 0.35 for c in range(4))


def test_ping_and_iperf_parsing():
    out = "50 packets transmitted, 47 received, 6% packet loss, time 2451ms\nrtt min/avg/max/mdev = 61.1/80.2/130.9/9.3 ms"
    assert parse_ping(out, 15) == {"rtt_avg_ms": 80.2, "rtt_min_ms": 61.1, "rtt_max_ms": 130.9,
                                   "rtt_mdev_ms": 9.3, "packet_loss_pct": 6.0}
    assert parse_ping("50 packets transmitted, 0 received, 100% packet loss", 15)["rtt_avg_ms"] == 15000
    iperf = json.dumps({"end": {"sum_received": {"bits_per_second": 4e6}, "sum_sent": {"retransmits": 7}}})
    assert parse_iperf(iperf, 10) == {"throughput_mbps": 4.0, "bw_utilization_pct": 60.0, "retransmits": 7}
    assert parse_iperf("", 10)["bw_utilization_pct"] == 100.0


def test_passive_utilisation():
    key = (SPEC.switch_dpids["s15"], SPEC.port("s15", "s14"))
    util = link_utilization_from(SPEC, {key: 0}, {key: 5_000_000}, 8.0)   # 5 MB in 8 s = 5 Mbps
    assert util[("s15", "s14")] == pytest.approx(0.5) and util[("s14", "s15")] == 0.0
    features = passive_features(util, ["s15", "s14", "s11"])
    assert features["path_max_link_util_pct"] == pytest.approx(50.0)
    assert features["path_mean_link_util_pct"] == pytest.approx(12.5)


def synthetic_dataset(path, rows_per_class=60, seed=0):
    """Feature rows that grow noisier and more congested with the class (for pipeline tests only)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(4):
        for _ in range(rows_per_class):
            load = rng.uniform(*CLASS_BANDS[c])
            rtt = 120 + 900 * max(load - 0.5, 0) + rng.normal(0, 15)
            thr = max(10 * (1 - load), 0.2) + rng.normal(0, 0.3)
            rows.append({"rtt_avg_ms": rtt, "rtt_min_ms": rtt * 0.8, "rtt_max_ms": rtt * 1.5,
                         "rtt_mdev_ms": 5 + 40 * load, "packet_loss_pct": max(0, 30 * (load - 0.8)),
                         "throughput_mbps": thr, "bw_utilization_pct": 100 * (1 - thr / 10),
                         "retransmits": int(rng.poisson(50 * load)), "hop_count": int(rng.integers(3, 8)),
                         "path_max_link_util_pct": 100 * min(load, 1.0) + rng.normal(0, 3),
                         "path_mean_link_util_pct": 40 * load, "traffic_class": c})
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("training")
    synthetic_dataset(tmp / "data.csv")
    args = type("Args", (), {"data": str(tmp / "data.csv"), "feature_set": "all",
                             "model_out": str(tmp / "best_model.joblib"), "results": str(tmp / "results")})
    best = train_models.train(args)
    return tmp, best


def test_training_compares_all_models_and_saves_the_best(trained):
    tmp, best = trained
    table = pd.read_csv(tmp / "results" / "model_comparison.csv")
    assert set(table["model"]) == {"Logistic Regression", "Decision Tree", "SVM", "XGBoost", "Random Forest",
                                   "Bagging", "AdaBoost", "Gradient Boosting"}
    assert {"accuracy", "precision", "f1"} <= set(table.columns)
    assert (tmp / "results" / "model_comparison.png").exists()
    classifier = CongestionClassifier.load(str(tmp / "best_model.joblib"))
    assert classifier.name == best and classifier.features == FEATURE_NAMES


class FakeInstaller:
    def __init__(self):
        self.installed = []

    def install_path(self, src_ip, dst_ip, path, prefix, priority=32768, ip_protos=None):
        self.installed.append((prefix, tuple(path), priority, ip_protos))
        return [prefix]

    def remove(self, names):
        pass


class FakeTelemetry:
    """Path i congested according to `loads[i]`."""
    def __init__(self, loads):
        self.spec, self.loads = SPEC, loads

    def link_utilization(self):
        return {}

    def measure(self, src, dst, path, link_util):
        load = self.loads[candidate_paths(SPEC).index(path)]
        thr = max(10 * (1 - load), 0.2)
        rtt = 120 + 900 * max(load - 0.5, 0)
        return {"rtt_avg_ms": rtt, "rtt_min_ms": rtt * 0.8, "rtt_max_ms": rtt * 1.5, "rtt_mdev_ms": 5 + 40 * load,
                "packet_loss_pct": max(0, 30 * (load - 0.8)), "throughput_mbps": thr,
                "bw_utilization_pct": 100 * (1 - thr / 10), "retransmits": 50 * load, "hop_count": len(path) - 1,
                "path_max_link_util_pct": 100 * min(load, 1.0), "path_mean_link_util_pct": 40 * load}


def test_ml_routing_installs_the_least_congested_candidate(trained):
    tmp, _ = trained
    classifier = CongestionClassifier.load(str(tmp / "best_model.joblib"))
    candidates = candidate_paths(SPEC)
    installer = FakeInstaller()
    decision, names = ml_routing(candidates, SENDER, RECEIVER, installer, FakeTelemetry([1.1, 0.9, 0.1, 0.6, 0.35]),
                                 classifier)
    assert decision.chosen == candidates[2] and names == ["geant2ml"]
    assert [s.predicted_class for s in decision.scored][2] == 0
    assert installer.installed[-1] == ("geant2ml", tuple(candidates[2]), 32768, None)
    assert all(entry[3] == [1, 6] for entry in installer.installed[:-1])     # probes pin ICMP/TCP only


def test_evaluate_writes_tables_and_figures(tmp_path):
    rng = np.random.default_rng(1)
    for k in range(2):
        for i, approach in enumerate(evaluate.APPROACHES):
            directory = tmp_path / "geant2_x" / f"scenario_{k}" / approach
            directory.mkdir(parents=True)
            json.dump({"approach": approach, "mean_rtt_ms": 300 - 50 * i, "ping_loss_pct": 5 - i,
                       "mean_throughput_mbps": 2 + i, "psnr_db": 25 + 4 * i, "ssim": 0.9 + 0.02 * i,
                       "frames_received": 4000}, open(directory / "summary.json", "w"))
            pd.DataFrame({"t_s": range(1, 61), "rtt_ms": rng.normal(300 - 50 * i, 10, 60),
                          "throughput_mbps": rng.normal(2 + i, 0.2, 60)}).to_csv(directory / "timeseries.csv", index=False)
            pd.DataFrame({"frame": range(200), "psnr_db": rng.normal(25 + 4 * i, 1, 200),
                          "ssim": rng.normal(0.9 + 0.02 * i, 0.01, 200)}).to_csv(directory / "frames.csv", index=False)
            json.dump({"chosen_path": ["s15", "s16"], "chosen_ground_truth": {"class": 3 - i, "class_name": "x"}},
                      open(directory / "decision.json", "w"))
    run = evaluate.find_runs(str(tmp_path))[0]
    summary = evaluate.evaluate(run)
    assert list(summary.index) == [evaluate.LABELS[a] for a in evaluate.APPROACHES]
    assert summary.loc["GEANT2 multi-class ML", "psnr_db"] == pytest.approx(37.0)
    for name in ("summary.png", "rtt_throughput.png", "video_quality.png", "summary_table.csv", "per_scenario.csv"):
        assert (tmp_path / "geant2_x" / "figures" / name).exists()
