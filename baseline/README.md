# AIRSDN baseline pipeline

A modular reimplementation of the paper's evaluation (İpek et al., *AIRSDN*, Computer Communications 240,
2025). It keeps the paper's scope:
- the 14-switch NSFNET only;
- hop-count candidates only;
- Logistic Regression only;
- three routing approaches compared.

| File | Role |
|---|---|
| `topology_builder.py` | NSFNET (14 switches, 21 links, 25 ms / 10 Mbps), the paper's 20 hosts and 9 pinned background flows |
| `path_discovery.py` | Stage 1: Yen's 5 shortest paths by hop count (same ranking as Floodlight) |
| `telemetry_and_model.py` | Stage 2: the 19 path features (ping, iperf3, NFStream), Logistic Regression, static flow rules, the routing algorithm of Table 2 |
| `multimedia_evaluator.py` | 60 s RTT and throughput monitoring, FFmpeg video streaming, PSNR/SSIM |
| `visualize_benchmarks.py` | Hop-count vs QoS-aware vs AIRSDN figures |
| `run_experiment.py` | Runs the three approaches on one network |

Compared approaches:
- **Hop-count:** Floodlight's default reactive routing.
- **QoS-aware:** Floodlight's reactive routing with the latency metric.
- **AIRSDN:** probes each of the 5 hop-count candidates, classifies its traffic level with Logistic
  Regression, and installs the path with the lowest predicted level (ties go to the earlier candidate).

## Running

```bash
./setup.sh                                       # once (Fedora): dependencies, Floodlight, .venv
.venv/bin/python train_models.py                 # once: trains models/logistic_regression_model.pkl
./start_floodlight.sh                            # separate terminal
sudo .venv/bin/python baseline/run_experiment.py
.venv/bin/python baseline/visualize_benchmarks.py results/baseline
.venv/bin/python -m pytest baseline/tests        # offline tests
```

A run takes about 17 minutes:
- `pingAll`: about 2 minutes
- three × (60 s RTT + 60 s throughput + 66 s video)
- AIRSDN probing: 5 paths × about 55 s

Results go to `results/baseline/nsfnet_<timestamp>/<approach>/`:
- `timeseries.csv`
- `frames.csv`
- `summary.json`
- `decision.json`: AIRSDN's candidates, 19 features, predicted levels and timings

Figures go to `<run>/figures/`.
