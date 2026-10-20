# GEANT2 multi-metric + multi-class routing (combined architecture)

Branch `geant2-multi-metric-advanced`. It combines two earlier branches:

- **GEANT2:** the topology, 4-class background traffic, the multi-class ensemble and its 500-row dataset.
- **improvements:** the two-stage pipeline, with multi-metric candidate discovery followed by ML selection.

Both of those branches are left unchanged. The code that this branch reuses was copied from them with
`git show`, so it has no merge history with them.

> **Topology size.** GEANT2 has **24 nodes and 37 links**: the RouteNet / DRL-GNN edge list, and the
> network the dataset was collected on. The request mentioned 30 nodes, but no GEANT2 variant used in
> the literature has 30. Using another size would also push the Stage 2 features outside the model's
> training data.

## Branch setup

```bash
git switch main
git switch -c geant2-multi-metric-advanced
# The dataset is reused from the GEANT2 branch, read-only:
git show GEANT2:geant2/data/geant2_dataset.csv > advanced/data/geant2_dataset.csv
```

## Architecture

```
               link telemetry (Floodlight port counters + LLDP latency, ~12 s)
                                     │
          ┌──────────────────────────▼──────────────────────────┐
Stage 1   │ path_discovery.py   top-K (5-10) paths, togglable:  │   ~1-6 ms
          │   hop | latency | bandwidth (widest) | composite    │
          │   composite = α·D + β·J + γ·(1/BW) + δ·L (normalised)│
          └──────────────────────────┬──────────────────────────┘
                                     │ K candidates, each with its composite cost
          ┌──────────────────────────▼──────────────────────────┐
Stage 2   │ telemetry_and_model.py                              │   probe ~10 s / candidate
          │   probe each candidate (ping + iperf3, 11 features) │   inference ~28 ms / decision
          │   ensemble (XGBoost, RF, GB, SVM) → class 0..3      │
          │   choose lowest class; tie → lowest composite cost  │
          └──────────────────────────┬──────────────────────────┘
                                     │ OpenFlow 1.3 static entries (Floodlight StaticEntryPusher)
                                     ▼
          multimedia_evaluator.py: H.264 stream Sender h15 → Receiver h23, per-second QoS + per-frame PSNR/SSIM
```

| Module | File | Contents |
|---|---|---|
| 1 | [topology_builder.py](topology_builder.py) | GEANT2 in Mininet and NetworkX (10 Mbit/s, 10 ms, 1,000-packet queues). Static ARP, IPv6 off. 4-class scenarios (EL / L / H / EH by link utilisation), driven by pinned iperf3 UDP flows on the core links. |
| 2 | [path_discovery.py](path_discovery.py) | Hop-bounded Yen's K-shortest paths over 4 metrics, a widest path by max-bottleneck Dijkstra, and the composite cost. Each call reports `compute_ms`. |
| 3 | [telemetry_and_model.py](telemetry_and_model.py) | Floodlight REST client, link telemetry snapshot, 11-feature path probe, model bundle loader, batched prediction, selection rule, flow installer. |
| 4 | [multimedia_evaluator.py](multimedia_evaluator.py) | FFmpeg H.264 transfer. Writes `timeseries.csv` (RTT, jitter, throughput, loss per second), `frames.csv` (PSNR and SSIM per frame) and `summary.json`. |
| 5 | [visualize_benchmarks.py](visualize_benchmarks.py) | Line plots (throughput, RTT, jitter), loss box plots, PSNR and SSIM bars, and Stage 1 + Stage 2 compute-latency bars. |
| – | [train_models.py](train_models.py) | Trains the 4 models and the soft-voting ensemble, and saves `models/multiclass_bundle.joblib`. |
| – | [run_experiment.py](run_experiment.py) | Live end-to-end benchmark and the offline timing benchmark. |
| – | [udp_probe.py](udp_probe.py) | UDP probe used to measure jitter and loss (RFC 3550). |

### The paradigms compared

| Paradigm | Stage 1 | Stage 2 | Role |
|---|---|---|---|
| `hop_count` | hop, K = 1 | – | legacy shortest path |
| `qos_aware` | latency on measured LLDP delay, K = 1 | – | QoS-aware routing |
| `hop_ml` | hop, K = 5 | ensemble | the approach from the GEANT2 branch (ablation) |
| `advanced` | composite (or `--stage1-method`), K = 5–10 | ensemble | this architecture |

## Prerequisites

The repository's `./setup.sh` installs everything except `imageio-ffmpeg` and `networkx`, which are now in
`requirements.txt`:

```bash
.venv/bin/pip install -r requirements.txt
```

Fedora's `ffmpeg-free` package can only decode H.264 through OpenH264. The quality scoring therefore uses the
FFmpeg build that ships with `imageio-ffmpeg`.

## Running

The first three steps need neither root nor Mininet:

```bash
.venv/bin/python -m pytest advanced/tests -q                 # 13 tests
.venv/bin/python advanced/topology_builder.py --preview      # topology + class balance of the scenarios
.venv/bin/python advanced/path_discovery.py --k 5            # Stage 1 on synthetic telemetry, all methods
.venv/bin/python advanced/train_models.py                    # retrain (bundle is committed)
.venv/bin/python advanced/run_experiment.py --benchmark-only # offline Stage 1 / Stage 2 timing
```

The live benchmark needs Floodlight and root:

```bash
./start_floodlight.sh                                         # terminal 1, leave running
sudo .venv/bin/python advanced/run_experiment.py --scenarios 3 --congest-default-path
.venv/bin/python advanced/visualize_benchmarks.py results/advanced \
    --benchmark results/advanced/decision_latency_benchmark.csv
```

Alternatively, `advanced/run_all.sh` runs every step in order and stops the machine from sleeping during the
run. `SCENARIOS=5 advanced/run_all.sh` changes the number of scenarios.

Useful options for `run_experiment.py`:

- `--stage1-method {hop,latency,bandwidth,composite}` sets the Stage 1 metric of the combined paradigm.
- `--k-paths 10`
- `--alpha/--beta/--gamma/--delta` set the composite weights (defaults 0.3 / 0.1 / 0.4 / 0.2).
- `--model XGBoost` picks one ensemble member instead of the full ensemble.
- `--paradigms advanced hop_count` runs only the listed paradigms.
- `--resume results/advanced/<run> --first-scenario k` continues an interrupted run.
- `--delay-range 5 25` uses heterogeneous link delays. These move the RTT features away from the training
  data.

Each scenario takes roughly 10 minutes: 4 paradigms, each with a telemetry snapshot, probing (ML paradigms) and a 66-second video plus quality analysis.

## Outputs

```
advanced/models/multiclass_bundle.joblib       trained models + ensemble + metrics
advanced/results/training/                     model_comparison.{csv,png}, confusion matrices
results/advanced/decision_latency_benchmark.csv
results/advanced/advanced_<timestamp>/
    run.json
    scenario_<k>/scenario.json                 link utilisation + class of every link
    scenario_<k>/<paradigm>/decision.json      candidates (ground-truth class), Stage-2 scores, timings
    scenario_<k>/<paradigm>/timeseries.csv     t_s, rtt_ms, jitter_ms, throughput_mbps, packet_loss_pct
    scenario_<k>/<paradigm>/frames.csv         per-frame PSNR / SSIM
    figures/                                   all plots + summary_table.csv, per_scenario.csv
```

`results/advanced/` is git-ignored because each run is large. The trained bundle and the training metrics
are committed.

## Results so far

The models were trained on the 500-row GEANT2 dataset, with 75 rows in the test split:

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| XGBoost | 0.973 | 0.974 |
| Random Forest | 0.973 | 0.974 |
| Gradient Boosting | 0.973 | 0.974 |
| SVM (RBF) | 0.933 | 0.931 |
| **Soft-voting ensemble** | 0.973 | 0.974 |

Offline decision cost on GEANT2, mean of 20 trials:

| | K = 5 | K = 10 |
|---|---|---|
| Stage 1 hop | 1.4 ms | 2.5 ms |
| Stage 1 latency | 2.9 ms | 5.4 ms |
| Stage 1 widest | 3.2 ms | 6.2 ms |
| Stage 1 composite | 2.2 ms | 4.0 ms |
| Stage 2 inference (batched) | 28.8 ms | 28.1 ms |

Stage 2 runs a single `predict_proba` call over all K candidates, so its cost does not grow with K. In a
live decision the dominant cost is measurement: about 12 s of link telemetry plus about 10 s of probing (ping, iperf3, queue drain) per
candidate. Computation is small next to that.

### Live benchmark (Mininet + Floodlight, 3 scenarios, `--congest-default-path`)

Run `results/advanced/advanced_20261020_202810` used scenario seeds 1000–1002. In every scenario, the 6 links of
the hop-count shortest path are forced to Extremely High (80–120 % load). Values are means over the
3 scenarios:

| Paradigm | Chosen path class (0–3) | Video loss | PSNR | SSIM | RTT | Jitter | Throughput |
|---|---|---|---|---|---|---|---|
| Hop-count | 3.00 | 99.6 % | 15.1 dB | 0.727 | all echoes lost | 24.1 ms | 0.02 Mbit/s |
| QoS-aware | 1.00 | 6.7 % | 31.3 dB | 0.928 | 505 ms | 12.0 ms | 4.33 Mbit/s |
| Hop candidates + ML | 1.00 | 6.8 % | 31.3 dB | 0.929 | 510 ms | 12.2 ms | 4.31 Mbit/s |
| **Combined (multi-metric + ML)** | **0.33** | **0.0 %** | **38.4 dB** | **0.998** | **161 ms** | 12.9 ms | **4.65 Mbit/s** |

The combined architecture delivered the video loss-free in all three scenarios.

- **Scenarios 1 and 2:** QoS-aware and hop + ML were also loss-free.
- **Scenario 0:** both of them stayed on a 6-hop High-class path and lost 20 % of the video. The combined
  method's composite Stage 1 offered a 7-hop path whose links were all Extremely Low, and Stage 2 chose it.

Live Stage 2 inference took 40–60 ms, above the offline figure because Mininet loads the CPU. Probing the
candidates took 81 s per decision for the combined method and 195 s for hop + ML: probes on congested
candidates wait for ping and iperf3 timeouts.
