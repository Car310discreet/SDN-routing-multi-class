"""
make_figures.py - figures for the combined-architecture technical report that the pipeline does not produce
itself, plus a committed copy of the reported run's figures and tables (results/ is git-ignored).

    figures/pipeline.png         the combined two-stage architecture (telemetry, Stage 1, Stage 2, forwarding)
    figures/scenario0_paths.png  GEANT2 in scenario 0: offered-load class of every link and the path each
                                 paradigm chose
    figures/run/                 the run's figures (visualize_benchmarks.py output)
    data/                        the run's summary_table.csv and per_scenario.csv

    .venv/bin/python advanced/report/make_figures.py          # -> advanced/report/
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
ADVANCED_DIR = os.path.dirname(REPORT_DIR)
PROJECT_DIR = os.path.dirname(ADVANCED_DIR)
sys.path.insert(0, ADVANCED_DIR)
from topology_builder import CLASS_NAMES, RECEIVER, SENDER, build_geant2, class_of, link_key  # noqa: E402
from visualize_benchmarks import INK, INK_2, MUTED, SURFACE, apply_style  # noqa: E402

logging.setLoggerClass(logging.Logger)   # importing Mininet replaces the logger class
RUN = os.path.join(PROJECT_DIR, "results", "advanced", "advanced_20261020_202810")
OUT = os.path.join(REPORT_DIR, "figures")
STAGE = {"io": ("#fdf3dc", "#c98500"), "one": ("#e8f1fb", "#2a78d6"), "two": ("#e6f6ef", "#1baf7a"),
         "out": ("#f3f2ee", "#898781")}
CLASS_COLORS = ["#e9cfa8", "#f0a85a", "#d9601f", "#962812"]     # ordinal: Extremely Low -> Extremely High
HIGHLIGHT = "#eda100"


def box(ax, x, y, w, h, title, body="", edge=INK_2, width=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor="#ffffff", edgecolor=edge, linewidth=width))
    ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=9.0, color=INK, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.062, body, ha="center", va="top", fontsize=7.6, color=INK_2, linespacing=1.35)


def group(ax, x, y, w, h, title, key):
    face, edge = STAGE[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                facecolor=face, edgecolor=edge, linewidth=1.3))
    ax.text(x + 0.012, y + h - 0.013, title, ha="left", va="top", fontsize=9.4, color=INK, fontweight="bold")


def arrow(ax, start, end, color=INK_2, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=color, linewidth=1.2,
                                 linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2))


def pipeline(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.8))
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.axis("off")

    group(ax, 0.01, 0.80, 0.98, 0.195, "Network state: LinkTelemetry (6 samples, 2.5 s apart, from Floodlight)", "io")
    box(ax, 0.03, 0.815, 0.28, 0.13, "Switch port counters",
        "rate per link → utilisation,\navailable bandwidth, loss")
    ax.text(0.33, 0.88, "+", ha="center", va="center", fontsize=14, color=INK_2)
    box(ax, 0.35, 0.815, 0.28, 0.13, "LLDP link latency", "delay = mean of the samples;\njitter = their std. dev.")
    arrow(ax, (0.63, 0.88), (0.69, 0.88))
    box(ax, 0.69, 0.815, 0.28, 0.13, "Annotated GEANT2 graph", "24 switches, 37 links:\ndelay, jitter, avail. BW, loss")

    group(ax, 0.01, 0.53, 0.98, 0.25,
          "Stage 1: multi-metric candidate discovery (top-K loopless paths, K = 5–10, hops ≤ shortest + 4)",
          "one")
    methods = [("Hop count", "link weight 1\n(GEANT2-branch\ncandidates)", INK_2, 1.1),
               ("Latency", "link weight =\nmeasured delay", INK_2, 1.1),
               ("Available bandwidth", "K widest paths\n(max-bottleneck\nDijkstra)", INK_2, 1.1),
               ("Composite (default)", "α·D/Dmax + β·J/Jmax\n+ γ/(B/Bmax) + δ·L\nα .3, β .1, γ .4, δ .2",
                HIGHLIGHT, 2.2)]
    for i, (title, body, edge, width) in enumerate(methods):
        box(ax, 0.03 + i * 0.24, 0.575, 0.22, 0.15, title, body, edge=edge, width=width)
    arrow(ax, (0.83, 0.815), (0.83, 0.78))
    ax.text(0.5, 0.543, "every candidate keeps its composite cost, whatever method produced it",
            ha="center", va="bottom", fontsize=7.6, color=MUTED, style="italic")

    group(ax, 0.01, 0.21, 0.98, 0.295, "Stage 2: multi-class ML selection", "two")
    box(ax, 0.03, 0.32, 0.29, 0.14, "Probe each candidate",
        "ICMP/TCP pinned at priority 40000:\n50 pings · 5 s iperf3 TCP · 2 s drain\n→ 11 features")
    box(ax, 0.355, 0.32, 0.29, 0.14, "Soft-voting ensemble",
        "XGBoost · Random Forest ·\nGradient Boosting · SVM (RBF)\none batched call → P(class 0–3)")
    box(ax, 0.68, 0.32, 0.29, 0.14, "Select",
        "lowest predicted class;\nties → lowest composite cost,\nthen Stage-1 rank")
    arrow(ax, (0.32, 0.39), (0.355, 0.39))
    arrow(ax, (0.645, 0.39), (0.68, 0.39))
    arrow(ax, (0.175, 0.53), (0.175, 0.505))
    face, edge = STAGE["io"]
    ax.add_patch(FancyBboxPatch((0.335, 0.225), 0.33, 0.07, boxstyle="round,pad=0.006,rounding_size=0.01",
                                facecolor=face, edgecolor=edge, linewidth=1.0, linestyle="--"))
    ax.text(0.5, 0.26, "offline: GEANT2 dataset (500 paths, 4 classes)\n→ train_models.py → model bundle",
            ha="center", va="center", fontsize=7.4, color=INK_2)
    arrow(ax, (0.5, 0.295), (0.5, 0.32), dashed=True)

    group(ax, 0.01, 0.01, 0.98, 0.18, "Forwarding and evaluation", "out")
    box(ax, 0.53, 0.025, 0.44, 0.12, "Install route",
        "Floodlight StaticEntryPusher, OpenFlow 1.3:\nall Sender ↔ Receiver IPv4 traffic, both directions")
    box(ax, 0.03, 0.025, 0.44, 0.12, "Multimedia evaluator",
        "H.264 stream h15 → h23: per-second RTT, jitter,\nthroughput, loss; per-frame PSNR and SSIM")
    arrow(ax, (0.825, 0.32), (0.825, 0.145))
    arrow(ax, (0.53, 0.085), (0.47, 0.085))
    fig.savefig(path)
    plt.close(fig)


def scenario_paths(path: str) -> None:
    spec = build_geant2()
    graph = spec.graph
    pos = nx.kamada_kawai_layout(graph)
    with open(os.path.join(RUN, "scenario_0", "scenario.json")) as handle:
        utilization = json.load(handle)["utilization"]
    load = {link_key(u, v): utilization[f"{link_key(u, v)[0]}-{link_key(u, v)[1]}"] for u, v in graph.edges}

    chosen = {}
    for paradigm in ("hop_count", "qos_aware", "hop_ml", "advanced"):
        with open(os.path.join(RUN, "scenario_0", paradigm, "decision.json")) as handle:
            decision = json.load(handle)
        chosen[paradigm] = (decision["chosen_path"], decision["chosen_ground_truth"])
    panels = [("Hop-count", *chosen["hop_count"]),
              ("QoS-aware and Hop candidates + ML", *chosen["qos_aware"]),
              ("Combined (multi-metric + ML)", *chosen["advanced"])]
    assert chosen["qos_aware"][0] == chosen["hop_ml"][0]

    src, dst = spec.hosts[SENDER].switch, spec.hosts[RECEIVER].switch
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    for ax, (title, route, truth) in zip(axes, panels):
        route_edges = list(zip(route, route[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=route_edges, ax=ax, edge_color=INK, width=7.5)
        for u, v in graph.edges:
            c = class_of(load[link_key(u, v)])
            nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], ax=ax, edge_color=CLASS_COLORS[c],
                                   width=3.0 if c else 2.2)
        others = [n for n in graph.nodes if n not in route]
        nx.draw_networkx_nodes(graph, pos, nodelist=others, ax=ax, node_color=SURFACE, edgecolors=MUTED,
                               linewidths=0.9, node_size=200)
        nx.draw_networkx_nodes(graph, pos, nodelist=route, ax=ax, node_color=SURFACE, edgecolors=INK,
                               linewidths=1.8, node_size=240)
        nx.draw_networkx_labels(graph, pos, labels={n: n[1:] for n in graph.nodes}, ax=ax, font_size=6.5,
                                font_color=INK)
        for switch, role in ((src, "Sender"), (dst, "Receiver")):
            ax.annotate(role, xy=pos[switch], xytext=(0, -13), textcoords="offset points", ha="center",
                        va="top", fontsize=7.8, color=INK, fontweight="bold")
        ax.set_title(title, fontsize=10.5, pad=16)
        ax.text(0, 1.0, f"{len(route) - 1} hops · bottleneck {truth['class_name']} (load "
                        f"{truth['bottleneck_load']:.2f})", transform=ax.transAxes, fontsize=8, color=MUTED,
                va="bottom")
        ax.axis("off")
        ax.margins(0.06)
    handles = [Line2D([0], [0], color=CLASS_COLORS[c], linewidth=3.5, label=CLASS_NAMES[c]) for c in range(4)]
    handles.append(Line2D([0], [0], color=INK, linewidth=6, label="chosen path"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.02),
               title="link class (offered background load)", title_fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def copy_run() -> None:
    os.makedirs(os.path.join(OUT, "run"), exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "data"), exist_ok=True)
    figures = os.path.join(RUN, "figures")
    for name in sorted(os.listdir(figures)):
        target = os.path.join(OUT, "run", name) if name.endswith(".png") else os.path.join(REPORT_DIR, "data", name)
        shutil.copyfile(os.path.join(figures, name), target)


def main() -> None:
    apply_style()
    os.makedirs(OUT, exist_ok=True)
    pipeline(os.path.join(OUT, "pipeline.png"))
    if os.path.isdir(RUN):
        scenario_paths(os.path.join(OUT, "scenario0_paths.png"))
        copy_run()
    else:
        print(f"{RUN} not found: kept the committed run figures and scenario map")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
