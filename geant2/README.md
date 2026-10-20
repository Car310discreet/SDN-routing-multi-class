# GEANT2: multi-class ML routing

This extends the AIRSDN baseline (`baseline/`, NSFNET, binary Logistic Regression) in three ways:
- a larger network: GEANT2, 24 switches and 37 links;
- four traffic classes instead of two;
- a comparison of classifiers, keeping the best one. By default these are the four most likely to do best
  on this data: XGBoost, Gradient Boosting, Random Forest and SVM. `--models all` adds Logistic
  Regression, Decision Tree, Bagging and AdaBoost.

The path selection keeps the baseline's structure: 5 hop-count candidates, classify each, take the
least congested. It runs on the same Mininet + Floodlight stack. The baseline's modules are reused unchanged:
- `FloodlightClient`, `FlowInstaller` and `PathProber`;
- Yen's algorithm;
- the video evaluator.

## Layout

```
geant2/
├── geant2_topo.py          GEANT2 in Mininet + NetworkX; Sender h15 (s15), Receiver h23 (s23)
├── visualize_topology.py   -> figures/geant2_topology.png, figures/geant2_candidates.png
├── traffic_gen.py          per-link iperf3 UDP background traffic in 4 classes, re-drawn every epoch
├── telemetry.py            path features: ping, iperf3, Floodlight port counters
├── collect_dataset.py      network + traffic + telemetry -> data/geant2_dataset.csv   (sudo)
├── ml_classifier.py        feature sets, loading and querying the saved model
├── train_models.py         top-4 classifiers compared -> models/best_model.joblib, results/training/
├── run_all.sh              the whole pipeline in one sudo command
├── controller_routing.py   Yen top-5 -> classify each -> install the least congested (Floodlight REST)
├── run_experiment.py       hop-count vs QoS-aware vs AIRSDN vs GEANT2 ML, with video   (sudo)
├── evaluate.py             PSNR, SSIM, RTT, throughput, packet loss: tables, bar and line charts
├── figures/                topology diagrams
├── tests/test_geant2.py    offline tests (no Mininet / Floodlight)
├── data/                   the collected dataset (created by collect_dataset.py)
├── models/                 the best model (created by train_models.py)
└── results/training/       model comparison (created by train_models.py)
```

Run outputs go to `results/geant2/<run>/`, which git ignores.

## Network

| | |
|---|---|
| Switches / links | 24 / 37 (GEANT2 graph from RouteNet / DRL-GNN) |
| Switch-switch links | 10 Mbps, 10 ms, 1000-packet queue (`--link-bw`, `--link-delay`) |
| Hosts | one per switch, `h<i>` = 10.0.0.<i+1> (`--hosts-per-switch` adds more); access links not shaped |
| Sender → Receiver | `h15` (s15) → `h23` (s23): 6 hops, the diameter of the network |

Hop-count candidates (Yen, K = 5):
- `s15-s14-s11-s8-s12-s19-s23`
- `s15-s16-s17-s8-s12-s19-s23`
- `s15-s16-s17-s18-s21-s22-s23`, which avoids the s8 hub
- two 7-hop variants.

## Traffic classes and labels

Every link carries a pinned iperf3 UDP flow in each direction. Its rate is set by the link's class:

| Class | Offered load / capacity |
|---|---|
| 0 Extremely Low | 0 – 25 % |
| 1 Low | 25 – 50 % |
| 2 High | 50 – 80 % |
| 3 Extremely High | 80 – 120 % (the queue fills and drops packets) |

A path's label is the class of its most loaded link, using the generator's ground truth, never a measured
value. Link classes are drawn so that the classes of 6-hop paths come out balanced. Run
`python geant2/traffic_gen.py --preview` to check: the candidates come out at 22–27 % per class. The
collector also caps every class at a quarter of the target, so the dataset is balanced.

There are 11 features: RTT avg/min/max/mdev, packet loss, TCP throughput, bandwidth utilisation, retransmits,
hop count, and the max and mean link utilisation along the path from Floodlight's port counters.
`train_models.py --feature-set active` drops the two controller features, like the baseline.

## Running

```bash
git switch GEANT2
.venv/bin/python -m pytest geant2/tests                     # offline checks
.venv/bin/python geant2/visualize_topology.py               # topology figures

./start_floodlight.sh                                       # separate terminal, keep running

# everything below in one command (logs to results/geant2/run_all.log; blocks system sleep while it runs):
sudo SAMPLES=500 geant2/run_all.sh                          # the run reported below, about 4 hours
sudo geant2/run_all.sh                                      # 4,000 rows, about 22 hours
sudo SAMPLES=200 SCENARIOS=1 geant2/run_all.sh              # smoke test, about 1 hour

# or step by step:
# 1. dataset: about 150-190 rows per hour; Ctrl+C and rerun to resume
sudo .venv/bin/python geant2/collect_dataset.py --samples 500

# 2. models: compare the top-4 classifiers, save the best (no sudo); --models all for all 8
.venv/bin/python geant2/train_models.py

# 3. routing + video evaluation, 3 traffic scenarios (about 22 minutes each); an interrupted run
#    continues with --resume results/geant2/<run> --first-scenario <k>
sudo .venv/bin/python geant2/run_experiment.py --scenarios 3

# 4. comparison figures and tables
.venv/bin/python geant2/evaluate.py results/geant2
```

Other useful commands:
- `sudo .venv/bin/python geant2/geant2_topo.py --cli` opens the network in the Mininet CLI.
- `sudo .venv/bin/python geant2/traffic_gen.py --epoch 60` runs fluctuating traffic on its own.
- `.venv/bin/python geant2/controller_routing.py` lists the candidates and compares them with Floodlight's
  `/wm/routing/paths` while the network is up.

## What is compared

`run_experiment.py` applies one seeded traffic scenario. All four approaches route the video under it,
one after another:
- **Hop-count** and **QoS-aware**: Floodlight's own routing, as in the baseline.
- **AIRSDN (baseline)**: the paper's approach unchanged. It probes the same 5 candidates (19 features),
  uses `models/logistic_regression_model.pkl` (binary, trained on NSFNET) and picks the lowest level.
- **GEANT2 multi-class ML**: picks the lowest of 4 predicted classes. Ties go to the lower
  probability-weighted class, then the shorter path.

Each route is measured the same way: 60 s ping (RTT and packet loss), 60 s iperf3 throughput, and the
66 s H.264 video (FFmpeg, UDP) scored with PSNR and SSIM. `decision.json` also records the ground-truth
class of every candidate and of the chosen path. That shows how well each approach avoided the
congested links, not only how the video came out.

Videos are scored with the static FFmpeg of `imageio-ffmpeg` (in `requirements.txt`). Fedora's
`ffmpeg-free` decodes H.264 only with OpenH264, which cannot decode a stream damaged by heavy loss.

## Results (20 October 2026)

All files are in `geant2/results/`: `training/`, `training_active/` and `evaluation/`. The dataset is
`geant2/data/geant2_dataset.csv`, and the models are in `geant2/models/`.

### Dataset

- 500 rows, 125 per class, measured on the live network over 66 epochs.
- 3 rows (0.6 %) carry a label that does not match the measurement: a background flow did not start, so
  Floodlight measured far less load than was set. They are kept.

### Models

Test set: 75 rows. The best model is chosen on validation F1.

| Model | Accuracy | Precision | F1 | Validation F1 |
|---|---|---|---|---|
| **Gradient Boosting** (saved, `best_model.joblib`) | 0.973 | 0.976 | 0.974 | **0.987** |
| XGBoost | 0.973 | 0.976 | 0.974 | 0.973 |
| Random Forest | 0.973 | 0.976 | 0.974 | 0.946 |
| SVM | 0.933 | 0.942 | 0.931 | 0.933 |

- The best three models misclassify the same 2 test rows, both mislabelled rows.
- Controller-measured link utilisation accounts for 82 % of Gradient Boosting's feature importance.
- Without the two controller features (`--feature-set active`), the tree ensembles still reach 0.973
  and SVM 0.907. XGBoost is best on validation; it is saved as `best_model_active.joblib`.

### Routing and video

3 seeded traffic scenarios, means over scenarios:

| Approach | PSNR (dB) | SSIM | RTT (ms) | Throughput (Mbps) | Ping loss (%) | Frames decoded | Chosen path class |
|---|---|---|---|---|---|---|---|
| Hop-count | 30.5 | 0.913 | 830 | 4.65 | 22.2 | 2,907 | Low, Extremely High, Extremely Low |
| QoS-aware | 24.8 | 0.858 | 514 | 3.18 | 20.0 | 3,667 | High, Low, Extremely Low |
| AIRSDN (baseline) | 38.4 | 0.998 | 120 | 6.69 | 0.0 | 3,968 | Low, Low, Extremely Low |
| **GEANT2 multi-class ML** | **38.4** | **0.998** | 133 | **7.17** | **0.0** | **3,968** | **Extremely Low ×3** |

Per approach:
- **GEANT2 multi-class ML:** chose the least loaded class every time. It had the highest throughput in
  every scenario. RTT was 20 ms higher in two scenarios because the quieter path had one more hop.
- **Hop-count:** sent the video over the congested path in scenario 1: PSNR 14.8 dB, 785 frames
  decoded, 2.3 s RTT.
- **QoS-aware:** picked an overloaded (High) path in scenario 0: PSNR 16.8 dB.
- **AIRSDN:** avoided congestion in all three scenarios, so its video quality equals the new model's.
  Its binary classifier cannot separate Extremely Low from Low, so it took Low paths and got lower
  throughput.

Decision time:
- The new model probes for 5 s per path: 1-2 min per decision.
- AIRSDN probes for 30 s per path, plus NFStream: 3.7-5.2 min per decision.

Prediction accuracy on the 15 candidates (3 scenarios × 5 paths):
- The 4-class model got 12 of 15 exactly right. The other 3 were each off by one class, at a class
  boundary.
- AIRSDN's binary model got 13 of 15 right, on its easier two-class task.

With 3 scenarios, these numbers show the behaviour; they are not a statistical result. Run more with
`--scenarios N`.
