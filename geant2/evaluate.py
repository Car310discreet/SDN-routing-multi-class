"""
evaluate.py - GEANT2 results: multi-class ML routing vs the baseline approaches.

Reads results/geant2/<run>/scenario_<k>/<approach>/{summary.json,timeseries.csv,frames.csv,decision.json}
(written by run_experiment.py) and writes into <run>/figures/:
    summary.png          bar charts: PSNR, SSIM, mean RTT, mean throughput, packet loss
                         (mean over scenarios; whiskers span min-max when there are several)
    rtt_throughput.png   line plots: RTT and throughput per second (mean over scenarios)
    video_quality.png    line plots: per-frame PSNR and SSIM (mean over scenarios, 60-frame rolling mean,
                         about 1 s of the ~61 fps test video)
    summary_table.csv    per approach: mean metrics and the ground-truth class of the chosen paths
    per_scenario.csv     every scenario x approach

The baseline paper's approach is "airsdn" (binary Logistic Regression on the same 5 candidates);
Floodlight's hop-count and QoS-aware routing are the reference controllers.

    .venv/bin/python geant2/evaluate.py results/geant2            # every run in the directory
    .venv/bin/python geant2/evaluate.py results/geant2/geant2_<timestamp>
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baseline"))
import matplotlib.pyplot as plt  # noqa: E402
from visualize_benchmarks import COLORS as BASELINE_COLORS  # noqa: E402  (also selects the Agg backend)
from visualize_benchmarks import GRID, INK, INK_2, MUTED, apply_style  # noqa: E402

log = logging.getLogger("evaluate")

APPROACHES = ["hop_count", "qos_aware", "airsdn", "geant2_ml"]
LABELS = {"hop_count": "Hop-count", "qos_aware": "QoS-aware", "airsdn": "AIRSDN (baseline)",
          "geant2_ml": "GEANT2 multi-class ML"}
SHORT_LABELS = {"hop_count": "Hop-\ncount", "qos_aware": "QoS-\naware", "airsdn": "AIRSDN\n(baseline)",
                "geant2_ml": "GEANT2\nML"}
# Categorical slots 1-4 of the validated palette; aqua and yellow are below 3:1 against the surface,
# so every bar carries a value label in ink and every line plot has a legend.
COLORS = {**BASELINE_COLORS, "geant2_ml": "#eda100"}
METRICS = [("psnr_db", "PSNR (dB)", "higher is better"), ("ssim", "SSIM", "higher is better"),
           ("mean_rtt_ms", "Mean RTT (ms)", "lower is better"),
           ("mean_throughput_mbps", "Mean throughput (Mbps)", "higher is better"),
           ("ping_loss_pct", "Packet loss (%)", "lower is better")]


def find_runs(path: str) -> list[str]:
    if glob.glob(os.path.join(path, "scenario_*")):
        return [path]
    runs = sorted(d for d in glob.glob(os.path.join(path, "*")) if glob.glob(os.path.join(d, "scenario_*")))
    if not runs:
        raise ValueError(f"{path}: no GEANT2 runs (scenario_* directories) found")
    return runs


def load_run(run_dir: str) -> dict:
    """{approach: {"summaries": [...], "timeseries": [df...], "frames": [df...], "decisions": [...]}}"""
    data = {}
    for scenario_dir in sorted(glob.glob(os.path.join(run_dir, "scenario_*")),
                               key=lambda d: int(d.rsplit("_", 1)[1])):
        scenario = int(scenario_dir.rsplit("_", 1)[1])
        for approach in APPROACHES:
            directory = os.path.join(scenario_dir, approach)
            summary_path = os.path.join(directory, "summary.json")
            if not os.path.exists(summary_path):
                continue
            entry = data.setdefault(approach, {"summaries": [], "timeseries": [], "frames": [], "decisions": []})
            entry["summaries"].append({"scenario": scenario, **json.load(open(summary_path))})
            for name in ("timeseries", "frames"):
                path = os.path.join(directory, f"{name}.csv")
                if os.path.exists(path):
                    entry[name].append(pd.read_csv(path))
            path = os.path.join(directory, "decision.json")
            entry["decisions"].append(json.load(open(path)) if os.path.exists(path) else {})
    if not data:
        raise ValueError(f"{run_dir}: no results")
    return data


def tables(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for approach, entry in data.items():
        for summary, decision in zip(entry["summaries"], entry["decisions"]):
            truth = decision.get("chosen_ground_truth") or {}
            rows.append({"scenario": summary["scenario"], "approach": approach,
                         **{key: summary.get(key) for key, _, _ in METRICS},
                         "frames_received": summary.get("frames_received"),
                         "chosen_path": "-".join(decision.get("chosen_path") or []),
                         "chosen_true_class": truth.get("class"), "chosen_true_class_name": truth.get("class_name")})
    per_scenario = pd.DataFrame(rows)
    numeric = [key for key, _, _ in METRICS] + ["frames_received", "chosen_true_class"]
    summary = per_scenario.groupby("approach")[numeric].agg(lambda s: pd.to_numeric(s, errors="coerce").mean())
    summary.insert(0, "scenarios", per_scenario.groupby("approach").size())
    summary = summary.reindex([a for a in APPROACHES if a in summary.index])
    summary.index = [LABELS[a] for a in summary.index]
    summary.index.name = "approach"
    return per_scenario, summary


def plot_summary(per_scenario: pd.DataFrame, path: str) -> None:
    approaches = [a for a in APPROACHES if a in set(per_scenario["approach"])]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.5 * len(METRICS), 4.2))
    for ax, (key, title, hint) in zip(axes, METRICS):
        values = [pd.to_numeric(per_scenario.loc[per_scenario["approach"] == a, key], errors="coerce").dropna()
                  for a in approaches]
        means = [v.mean() if len(v) else np.nan for v in values]
        x = np.arange(len(approaches))
        ax.bar(x, np.nan_to_num(means), width=0.66, color=[COLORS[a] for a in approaches], edgecolor="none")
        for i, v in enumerate(values):
            if len(v) > 1:
                ax.vlines(i, v.min(), v.max(), color=INK_2, linewidth=1.2)
            label = "n/a" if np.isnan(means[i]) else f"{means[i]:.3f}" if key == "ssim" else f"{means[i]:.1f}"
            ax.annotate(label, xy=(i, 0 if np.isnan(means[i]) else max(means[i], v.max() if len(v) else 0)),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x, [SHORT_LABELS[a] for a in approaches], fontsize=8)
        ax.set_title(title, pad=18)
        ax.text(0, 1.02, hint, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
        ax.set_ylim(bottom=0)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(path)
    plt.close(fig)


def _mean_series(frames: list[pd.DataFrame], x: str, y: str) -> pd.Series:
    series = [df.set_index(x)[y] for df in frames if y in df]
    if not series:
        return pd.Series(dtype=float)
    return pd.concat(series, axis=1).apply(pd.to_numeric, errors="coerce").mean(axis=1)


def plot_lines(data: dict, path: str, x: str, panels: list[tuple[str, str]], xlabel: str, rolling: int = 1) -> None:
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3.0 * len(panels)), sharex=True)
    for ax, (column, title) in zip(np.atleast_1d(axes), panels):
        for approach in APPROACHES:
            if approach not in data:
                continue
            key = "timeseries" if x == "t_s" else "frames"
            series = _mean_series(data[approach][key], x, column)
            if series.empty:
                continue
            if rolling > 1:
                series = series.rolling(rolling, min_periods=1).mean()
            ax.plot(series.index, series.values, color=COLORS[approach], label=LABELS[approach])
        ax.set_title(title)
        ax.grid(axis="x", color=GRID)
    np.atleast_1d(axes)[-1].set_xlabel(xlabel)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def evaluate(run_dir: str) -> pd.DataFrame:
    data = load_run(run_dir)
    per_scenario, summary = tables(data)
    out = os.path.join(run_dir, "figures")
    os.makedirs(out, exist_ok=True)
    per_scenario.to_csv(os.path.join(out, "per_scenario.csv"), index=False)
    summary.round(4).to_csv(os.path.join(out, "summary_table.csv"))
    apply_style()
    plot_summary(per_scenario, os.path.join(out, "summary.png"))
    plot_lines(data, os.path.join(out, "rtt_throughput.png"), "t_s",
               [("rtt_ms", "RTT (ms)"), ("throughput_mbps", "Throughput (Mbps)")],
               "Time (s)")
    plot_lines(data, os.path.join(out, "video_quality.png"), "frame",
               [("psnr_db", "PSNR per frame (dB, 60-frame rolling mean)"), ("ssim", "SSIM per frame (60-frame rolling mean)")],
               "Frame", rolling=60)
    print(f"\n{run_dir}")
    print(summary.round(3).to_string())
    log.info("Figures in %s", out)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results", "geant2"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        for run_dir in find_runs(args.path):
            evaluate(run_dir)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
