# Baseline results (NSFNET)

Output of `baseline/run_experiment.py` on Fedora 43 (Mininet 2.3.1b4, Open vSwitch 3.6, Floodlight
master). It comes from a single run with the paper's background traffic.

| Approach | Mean RTT | Mean throughput | PSNR | SSIM |
|---|---|---|---|---|
| Hop-count (Floodlight default) | 1830 ms | 0.21 Mbps | 20.40 dB | 0.874 |
| QoS-aware (Floodlight latency) | 1896 ms | 6.64 Mbps | 38.11 dB | 0.997 |
| AIRSDN (hop-count candidates + LR) | **351 ms** | 6.52 Mbps | 38.11 dB | 0.997 |

Paper values: hop-count 2063 ms / 21.45 dB / 0.927, QoS-aware 1039 ms / 21.57 dB / 0.929,
AIRSDN 354.05 ms / 37.27 dB / 0.997.

**AIRSDN's choice matches the paper.** It classified candidates 0, 1, 2 and 4 as high traffic and
chose candidate 3, s0-s3-s4-s5-s12-s9, the path in the paper's Fig. 15.

**Floodlight's QoS-aware routing is not consistent between flows.** It routes each flow when the
flow starts, using noisy LLDP latency estimates. In this run the ping took a congested path while
the iperf3 and video flows took a clean one.

- `rtt_throughput.png`: RTT and throughput over 60 s (paper Figs. 16–17)
- `summary.png`: mean RTT, mean throughput, PSNR, SSIM
- `video_quality.png`: per-frame PSNR and SSIM
- `summary_table.csv`
