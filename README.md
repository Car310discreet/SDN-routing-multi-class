# cn_improvement

Re-implementation of *AIRSDN: AI based routing in software-defined networks for multimedia
traffic transmission* (İpek et al., Computer Communications 240, 2025) on Mininet + Floodlight.

## Setup (once)

```bash
./setup.sh
```

This installs `iproute-tc`, Open vSwitch, iperf3, ffmpeg and Python 3.11, then creates `.venv`
from `requirements.txt`. It also builds Mininet's `mnexec`, and downloads JDK 8 and Maven to `~/opt`.
Last, it builds Floodlight master into `~/floodlight` (v1.2 lacks the `/wm/routing/paths` and
`/wm/routing/metric` REST APIs used here).

## Train the classifier

```bash
.venv/bin/python train_models.py
```

This uses a 70/15/15 split and trains XGBoost, Random Forest, SVM, KNN, AdaBoost and Logistic Regression.
It prints the Table 5 / Table 6 metrics and saves `models/logistic_regression_model.pkl`.

## Run the experiments

Floodlight must be running in its own terminal:

```bash
./start_floodlight.sh
```

Then run each experiment from this directory. Static flows left by a previous run are cleared at start-up.

```bash
# 1. Default Floodlight routing (hop count)
sudo .venv/bin/python experiment/qos-experiment.py hopcount

# 2. Floodlight QoS-aware routing (latency; or "utilization")
sudo .venv/bin/python experiment/qos-experiment.py latency

# 3. Proposed AI-based routing
xhost +SI:localuser:root   # only needed for the ffplay preview window
sudo DISPLAY=:0 .venv/bin/python experiment/ai-based-routing-algorithm-experiment.py

# 3b. Same, re-running the routing algorithm every 30 s during the video (paper Section 3)
sudo .venv/bin/python experiment/ai-based-routing-algorithm-experiment.py --reroute-interval 30
```

Re-routing is off by default. A probe cycle (5 paths × ~45 s) is longer than the 66 s video, and each
probe saturates the client's access link, so it lowers the PSNR/SSIM of the video it protects.
Probe flows only match ICMP/TCP, so they never move the UDP video stream.

Each run logs PSNR/SSIM. It writes the 60 s RTT and throughput measurements to
`output_<metric>_ping.txt` / `output_<metric>_throughput.txt`, or `output_ai_*.txt` for the AI run.

## Clean up

```bash
sudo bash clear_resources/deleteFile.sh records/input.ts
sudo bash clear_resources/mininet.sh
```
