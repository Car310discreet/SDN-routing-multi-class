"""
visualize_topology.py - diagrams of the GEANT2 network (no root needed).

    geant2_topology.png     24 switches, 37 links, Sender and Receiver hosts
    geant2_candidates.png   the 5 hop-count candidate paths Sender -> Receiver (Yen)

    .venv/bin/python geant2/visualize_topology.py            # -> geant2/figures/
"""
from __future__ import annotations

import argparse
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geant2_topo import LINK_BW_MBPS, LINK_DELAY, RECEIVER, SENDER, build_geant2  # noqa: E402
from path_discovery import yen_hop_count  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from visualize_benchmarks import AXIS, INK, INK_2, MUTED, SURFACE, apply_style  # noqa: E402  (selects Agg)

GEANT2_DIR = os.path.dirname(os.path.abspath(__file__))
ROLE_COLORS = {"Sender": "#2a78d6", "Receiver": "#eb6834"}     # categorical slots 1 and 2
PATH_COLOR = "#2a78d6"


def layout(graph: nx.Graph) -> dict:
    return nx.kamada_kawai_layout(graph)


def draw_base(ax, graph, pos, edge_color=AXIS, width=1.4):
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_color, width=width)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=SURFACE, edgecolors=INK_2, linewidths=1.2, node_size=430)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_color=INK)


def host_position(pos, graph, switch):
    """Place a host just outside its switch, away from the centre of the drawing."""
    x, y = pos[switch]
    cx = sum(p[0] for p in pos.values()) / len(pos)
    cy = sum(p[1] for p in pos.values()) / len(pos)
    dx, dy = x - cx, y - cy
    norm = max((dx ** 2 + dy ** 2) ** 0.5, 1e-6)
    return x + 0.22 * dx / norm, y + 0.22 * dy / norm


def plot_topology(spec, path: str) -> None:
    graph, pos = spec.graph, layout(spec.graph)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    draw_base(ax, graph, pos)
    for role, host in (("Sender", SENDER), ("Receiver", RECEIVER)):
        info = spec.hosts[host]
        hx, hy = host_position(pos, graph, info.switch)
        ax.plot([pos[info.switch][0], hx], [pos[info.switch][1], hy], color=ROLE_COLORS[role], linewidth=2, zorder=1)
        ax.scatter([hx], [hy], marker="s", s=260, color=ROLE_COLORS[role], edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{role}\n{host} ({info.ip})", xy=(hx, hy), xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, color=INK)
    hops = nx.shortest_path_length(graph, spec.hosts[SENDER].switch, spec.hosts[RECEIVER].switch)
    ax.set_title(f"GEANT2: {graph.number_of_nodes()} switches, {graph.number_of_edges()} links "
                 f"({LINK_BW_MBPS} Mbps, {LINK_DELAY.replace('ms', ' ms')} each)")
    ax.text(0, -0.05, f"Sender and Receiver are {hops} hops apart (the network diameter). Every switch also has "
                    f"a host for background traffic (not drawn).", transform=ax.transAxes, fontsize=8, color=MUTED,
            va="top")
    ax.axis("off")
    ax.margins(0.08)
    fig.savefig(path)
    plt.close(fig)


def plot_candidates(spec, path: str) -> None:
    graph, pos = spec.graph, layout(spec.graph)
    src, dst = spec.hosts[SENDER].switch, spec.hosts[RECEIVER].switch
    candidates = yen_hop_count(graph, src, dst, 5)
    fig, axes = plt.subplots(1, len(candidates), figsize=(4.0 * len(candidates), 4.0))
    for i, (ax, candidate) in enumerate(zip(axes, candidates)):
        draw_base(ax, graph, pos, edge_color="#e1e0d9", width=1.0)
        edges = list(zip(candidate, candidate[1:]))
        nx.draw_networkx_edges(graph, pos, edgelist=edges, ax=ax, edge_color=PATH_COLOR, width=3.0)
        nx.draw_networkx_nodes(graph, pos, nodelist=candidate, ax=ax, node_color=SURFACE, edgecolors=PATH_COLOR,
                               linewidths=2.0, node_size=430)
        nx.draw_networkx_labels(graph, pos, labels={n: n for n in candidate}, ax=ax, font_size=8, font_color=INK)
        ax.set_title(f"Candidate {i}: {len(candidate) - 1} hops", fontsize=10)
        ax.text(0, -0.02, "-".join(candidate), transform=ax.transAxes, fontsize=7.5, color=INK_2, va="top")
        ax.axis("off")
    fig.suptitle(f"Top-5 hop-count paths {SENDER} ({src}) -> {RECEIVER} ({dst}), Yen's algorithm",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(GEANT2_DIR, "figures"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    apply_style()
    spec = build_geant2()
    plot_topology(spec, os.path.join(args.out, "geant2_topology.png"))
    plot_candidates(spec, os.path.join(args.out, "geant2_candidates.png"))
    print(f"Wrote geant2_topology.png and geant2_candidates.png to {args.out}")


if __name__ == "__main__":
    main()
