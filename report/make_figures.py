"""
make_figures.py - figures for the GEANT2 technical report that are not produced by the pipeline itself.

    architecture.png       system architecture (data plane, controller, measurement/learning, evaluation)
    routing_pipeline.png   the per-decision routing pipeline
    model_comparison.png   all eight classifiers, the deployed model highlighted
    scenario1_*.png        per-second RTT/throughput and per-frame PSNR/SSIM of the congested scenario 1
    metric_<name>.png      one bar chart per evaluation metric (PSNR, SSIM, RTT, throughput, packet loss),
                           from geant2/results/evaluation/per_scenario.csv

    .venv/bin/python report/make_figures.py          # -> report/figures/
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "geant2"))
sys.path.append(os.path.join(PROJECT_DIR, "baseline"))
from evaluate import APPROACHES, COLORS, SHORT_LABELS  # noqa: E402
from visualize_benchmarks import GRID, INK, INK_2, MUTED, SURFACE, apply_style  # noqa: E402

OUT = os.path.join(PROJECT_DIR, "report", "figures")
PER_SCENARIO = os.path.join(PROJECT_DIR, "geant2", "results", "evaluation", "per_scenario.csv")

# Light tints for diagram layers (one hue per layer, text stays in ink colours)
LAYER = {"data": "#e8f1fb", "control": "#fdeee6", "ml": "#e6f6ef", "eval": "#fdf3dc"}
EDGE = {"data": "#2a78d6", "control": "#eb6834", "ml": "#1baf7a", "eval": "#c98500"}


def box(ax, x, y, w, h, title, body="", face="#ffffff", edge=INK_2, title_size=9.5, body_size=7.8, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=face, edgecolor=edge, linewidth=1.1))
    if body:
        ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=title_size, color=INK,
                fontweight="bold" if bold else "normal")
        ax.text(x + w / 2, y + h - 0.058, body, ha="center", va="top", fontsize=body_size, color=INK_2,
                linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, color=INK,
                fontweight="bold" if bold else "normal")


def layer(ax, x, y, w, h, name, key):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                facecolor=LAYER[key], edgecolor=EDGE[key], linewidth=1.3))
    ax.text(x + 0.012, y + h - 0.012, name, ha="left", va="top", fontsize=9, color=INK, fontweight="bold")


def arrow(ax, start, end, label="", both=False, style="-|>", color=INK_2, label_offset=(0, 0), dashed=False,
          rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=("<|-|>" if both else style), mutation_scale=11,
                                 color=color, linewidth=1.2, linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        mx, my = (start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.4, color=INK_2,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))


def architecture(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.6))
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.axis("off")

    # Measurement and learning (top left)
    layer(ax, 0.01, 0.70, 0.70, 0.29, "Measurement and learning", "ml")
    ml = [("Traffic generator", "traffic_gen.py\n74 pinned iperf3\nUDP flows, 4 load\nclasses per epoch"),
          ("Telemetry", "telemetry.py\nping ×50, iperf3 5 s,\nFloodlight port\ncounters: 11 features"),
          ("Dataset", "collect_dataset.py\n500 paths,\n125 per class,\nground-truth labels"),
          ("Model training", "train_models.py\n8 classifiers,\nselect on val. F1:\nGradient Boosting")]
    for i, (title, body) in enumerate(ml):
        box(ax, 0.03 + i * 0.17, 0.73, 0.15, 0.21, title, body, title_size=8.8)
        if i:
            arrow(ax, (0.03 + i * 0.17 - 0.02, 0.835), (0.03 + i * 0.17, 0.835))

    # Evaluation (right column)
    layer(ax, 0.73, 0.37, 0.26, 0.62, "Evaluation", "eval")
    box(ax, 0.75, 0.76, 0.22, 0.18, "Network metrics", "ping 60 s: RTT, loss\niperf3 60 s: throughput")
    box(ax, 0.75, 0.56, 0.22, 0.18, "Video quality", "FFmpeg H.264, 66 s, UDP\nSender to Receiver\nPSNR and SSIM")
    box(ax, 0.75, 0.40, 0.22, 0.14, "Comparison", "evaluate.py\ntables and charts")
    arrow(ax, (0.86, 0.76), (0.86, 0.74))
    arrow(ax, (0.86, 0.56), (0.86, 0.54))

    # Control plane (middle left), with a free lane at x = 0.36 between the two boxes
    layer(ax, 0.01, 0.37, 0.70, 0.30, "Control plane", "control")
    box(ax, 0.03, 0.40, 0.29, 0.22, "Routing controller",
        "controller_routing.py\nYen top-5 by hop count;\nprobe, classify, pick\nlowest class")
    box(ax, 0.40, 0.40, 0.29, 0.22, "Floodlight controller",
        "REST: StaticEntryPusher,\nport statistics, links,\nrouting metric;\nreactive forwarding")
    arrow(ax, (0.32, 0.45), (0.40, 0.45), both=True)
    ax.text(0.338, 0.465, "REST", ha="center", va="bottom", fontsize=7.4, color=INK_2)

    # Data plane (bottom)
    layer(ax, 0.01, 0.01, 0.98, 0.33, "Data plane (Mininet)", "data")
    box(ax, 0.03, 0.04, 0.57, 0.24, "GEANT2: 24 Open vSwitch switches, 37 links",
        "10 Mbps, 10 ms, 1,000-packet queues per link\nOpenFlow 1.3; IPv6 off; static ARP\nbackground flows pinned per link")
    box(ax, 0.63, 0.165, 0.17, 0.115, "Sender h15", "on s15, video source", edge=EDGE["data"], title_size=9)
    box(ax, 0.63, 0.04, 0.17, 0.115, "Receiver h23", "on s23, video sink", edge=EDGE["data"], title_size=9)
    box(ax, 0.82, 0.04, 0.15, 0.24, "Other hosts", "h0 to h23,\none per switch:\nload endpoints,\nprobe pairs", title_size=9)

    # Cross-layer arrows, each in its own lane
    arrow(ax, (0.62, 0.73), (0.20, 0.62), color=EDGE["ml"])
    ax.text(0.47, 0.693, "trained model", ha="center", va="center", fontsize=7.4, color=INK_2,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=LAYER["ml"], edgecolor="none"))
    arrow(ax, (0.36, 0.73), (0.36, 0.28), color=EDGE["ml"], dashed=True)
    ax.text(0.352, 0.355, "load and probes (iperf3, ping)", ha="right", va="center", fontsize=7.4, color=INK_2)
    arrow(ax, (0.545, 0.40), (0.545, 0.28), both=True, color=EDGE["control"])
    ax.text(0.555, 0.385, "OpenFlow", ha="left", va="center", fontsize=7.4, color=INK_2)
    arrow(ax, (0.77, 0.37), (0.77, 0.28), color=EDGE["eval"], dashed=True)
    ax.text(0.78, 0.315, "stream and measure", ha="left", va="center", fontsize=7.4, color=INK_2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def routing_pipeline(path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(-0.005, 1.005)
    ax.axis("off")
    # Row A: candidates and passive telemetry
    box(ax, 0.10, 0.80, 0.30, 0.18, "1  Candidate paths", "Yen's K-shortest paths, K = 5,\nhop-count cost")
    box(ax, 0.60, 0.80, 0.30, 0.18, "2  Passive telemetry", "Floodlight port counters,\nread once over 3 s")
    arrow(ax, (0.40, 0.89), (0.60, 0.89))
    # Row B: per-candidate probing loop
    ax.add_patch(FancyBboxPatch((0.01, 0.31), 0.98, 0.43, boxstyle="round,pad=0.006,rounding_size=0.02",
                                facecolor=LAYER["ml"], edgecolor=EDGE["ml"], linewidth=1.2, linestyle="--"))
    ax.text(0.025, 0.725, "3  For each of the 5 candidates", ha="left", va="top", fontsize=9.5, color=INK,
            fontweight="bold")
    steps = [("Pin probe traffic", "ICMP/TCP static entries,\npriority 40000"),
             ("Measure", "ping ×50, iperf3 5 s:\n11 features"),
             ("Remove entries", "the video flow is never\nmoved by probes"),
             ("Classify", "Gradient Boosting:\nclass and probabilities")]
    for i, (title, body) in enumerate(steps):
        x = 0.03 + i * 0.245
        box(ax, x, 0.43, 0.205, 0.22, title, body, title_size=9)
        if i:
            arrow(ax, (x - 0.04, 0.54), (x, 0.54))
    arrow(ax, (0.85, 0.43), (0.13, 0.43), rad=-0.12, dashed=True)
    ax.text(0.49, 0.345, "next candidate", ha="center", va="center", fontsize=7.6, color=INK_2,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=LAYER["ml"], edgecolor="none"))
    arrow(ax, (0.75, 0.80), (0.75, 0.74))
    # Row C: select and install
    box(ax, 0.10, 0.02, 0.30, 0.22, "4  Select", "lowest predicted class;\nties: lower expected class,\nthen fewer hops")
    box(ax, 0.60, 0.02, 0.30, 0.22, "5  Install route", "bidirectional static entries\nfor all Sender and Receiver\ntraffic")
    arrow(ax, (0.25, 0.31), (0.25, 0.24))
    ax.text(0.26, 0.275, "after all 5", ha="left", va="center", fontsize=7.6, color=INK_2)
    arrow(ax, (0.40, 0.13), (0.60, 0.13))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scenario_timeseries(run_dir: str, scenario: int, path_net: str, path_video: str) -> None:
    """Per-second RTT/throughput and per-frame PSNR/SSIM of one scenario (no averaging across scenarios)."""
    from evaluate import LABELS, LINE_STYLE, PSNR_CAP_DB
    base = os.path.join(run_dir, f"scenario_{scenario}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for approach in APPROACHES:
        t = pd.read_csv(os.path.join(base, approach, "timeseries.csv"))
        style = LINE_STYLE.get(approach, {})
        rtt = pd.to_numeric(t["rtt_ms"], errors="coerce")
        lost = rtt.isna().any()   # lost pings break the line; markers keep isolated replies visible
        axes[0].plot(t["t_s"], rtt, color=COLORS[approach], label=LABELS[approach],
                     **({**style, "marker": "o", "markersize": 3.5} if lost else style))
        thr = pd.to_numeric(t["throughput_mbps"], errors="coerce")
        thr_style = dict(style)
        if thr.isna().all():      # iperf3 never connected over this path: it delivered nothing
            thr = pd.Series(0.0, index=t.index)
            if approach == "hop_count":   # would hide exactly under QoS-aware's zero line
                thr_style.update(linestyle=(0, (1, 2)), zorder=5, linewidth=2.6)
        axes[1].plot(t["t_s"], thr, color=COLORS[approach], label=LABELS[approach], **thr_style)
    axes[0].set_title("RTT (ms)")
    axes[1].set_title("Throughput (Mbps); 0 where iperf3 could not connect")
    axes[1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path_net)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for approach in APPROACHES:
        frames = pd.read_csv(os.path.join(base, approach, "frames.csv"))
        style = LINE_STYLE.get(approach, {})
        psnr = pd.to_numeric(frames["psnr_db"], errors="coerce").clip(upper=PSNR_CAP_DB).rolling(60, min_periods=1).mean()
        ssim = pd.to_numeric(frames["ssim"], errors="coerce").rolling(60, min_periods=1).mean()
        axes[0].plot(frames["frame"], psnr, color=COLORS[approach], label=LABELS[approach], **style)
        axes[1].plot(frames["frame"], ssim, color=COLORS[approach], label=LABELS[approach], **style)
    axes[0].set_title(f"PSNR per frame (dB, 60-frame rolling mean; identical frames at {PSNR_CAP_DB:.0f} dB)")
    axes[1].set_title("SSIM per frame (60-frame rolling mean)")
    axes[1].set_xlabel("Frame")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path_video)
    plt.close(fig)


METRIC_CHARTS = [
    ("psnr_db", "PSNR (dB)", "higher is better", "{:.1f}"),
    ("ssim", "SSIM", "higher is better", "{:.3f}"),
    ("mean_rtt_ms", "Mean RTT (ms)", "lower is better", "{:.0f}"),
    ("mean_throughput_mbps", "Mean throughput (Mbps)", "higher is better", "{:.2f}"),
    ("ping_loss_pct", "Packet loss (%)", "lower is better", "{:.1f}"),
]


def metric_chart(df: pd.DataFrame, key: str, title: str, hint: str, fmt: str, path: str) -> None:
    approaches = [a for a in APPROACHES if a in set(df["approach"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(approaches))
    for i, approach in enumerate(approaches):
        values = df.loc[df["approach"] == approach].sort_values("scenario")[key].astype(float)
        mean = values.mean()
        ax.bar(i, mean, width=0.62, color=COLORS[approach], edgecolor="none", zorder=2)
        ax.vlines(i, values.min(), values.max(), color=INK_2, linewidth=1.2, zorder=3)
        ax.scatter(np.full(len(values), i), values, s=22, color=SURFACE, edgecolors=INK, linewidths=1.0, zorder=4)
        ax.annotate(fmt.format(mean), xy=(i, max(mean, values.max())), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [SHORT_LABELS[a] for a in approaches], fontsize=8.5)
    ax.set_title(title, pad=18)
    ax.text(0, 1.02, f"{hint} · bar = mean of 3 scenarios, dots = scenarios, line = range", transform=ax.transAxes,
            fontsize=7.6, color=MUTED, va="bottom")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, df[key].astype(float).max() * 1.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    apply_style()
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = GRID
    architecture(os.path.join(OUT, "architecture.png"))
    routing_pipeline(os.path.join(OUT, "routing_pipeline.png"))
    from train_models import plot_comparison
    table = pd.read_csv(os.path.join(PROJECT_DIR, "geant2", "results", "training_all8", "model_comparison.csv"))
    plot_comparison(table, "Gradient Boosting", os.path.join(OUT, "model_comparison.png"),
                    note="Test-set scores of all eight classifiers (75 rows); blue: Gradient Boosting, the deployed "
                         "model (validation F1 0.987, tied with Bagging and Decision Tree)")
    run_dir = os.path.join(PROJECT_DIR, "results", "geant2", "geant2_20261020_172213")
    if os.path.isdir(run_dir):            # raw per-second / per-frame data (results/ is not in git)
        scenario_timeseries(run_dir, 1, os.path.join(OUT, "scenario1_network.png"),
                            os.path.join(OUT, "scenario1_video.png"))
    df = pd.read_csv(PER_SCENARIO)
    for key, title, hint, fmt in METRIC_CHARTS:
        metric_chart(df, key, title, hint, fmt, os.path.join(OUT, f"metric_{key}.png"))
    print(f"Figures in {OUT}")


if __name__ == "__main__":
    main()
