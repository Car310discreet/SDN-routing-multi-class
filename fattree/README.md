# Fat-tree (k=4): the second evaluation network

The GEANT2 multi-class pipeline is run on a network of a different kind: a k=4 fat-tree data-centre fabric.
Only the topology changes. The four traffic classes, the probes, the 11 features and the link settings
(10 Mbps, 10 ms, 1000-packet queue) all stay as on GEANT2.

## Why a fat-tree

Its switch graph differs from GEANT2's on every structural measure:

| | GEANT2 | Fat-tree k=4 |
|---|---|---|
| Switches / links | 24 / 37 | 20 / 32 |
| Degree | 2–8, one hub (s8) | 4 (core, aggregation), 2 (edge) |
| Diameter | 6 | 4 |
| Algebraic connectivity | 0.31 | 0.76 |
| Max betweenness centrality | 0.52 | 0.14 (no hub) |
| Sender → Receiver candidates (hops) | 6, 6, 6, 7, 7 | 4, 4, 4, 4, 6 |

Four of the five candidates have the same length, each through a different core switch. Hop count therefore
says almost nothing about which candidate is best, and the classifier has to rely on the load features.

## Network

| | |
|---|---|
| Core | s0–s3 |
| Aggregation | s4–s11, pod p: s(4+2p), s(5+2p) |
| Edge | s12–s19, pod p: s(12+2p), s(13+2p) |
| Hosts | one per switch, `h<i>` = 10.0.0.<i+1> on s<i> |
| Sender → Receiver | `h12` (s12, pod 0) → `h18` (s18, pod 3) |

The traffic generator pins one background flow onto every link, between hosts on its two ends. That is
why the core and aggregation switches also have a host. Those hosts only carry background traffic. The
Sender, the Receiver and the random extra probe pairs are all on edge switches.

The class balance targets 4-hop paths (`--path-hops 4`). The candidates come out at 23–27 % per class.

## Setup (Ubuntu 24.04)

The system packages are already installed on this machine: Open vSwitch, Mininet, iperf3, ffmpeg and
OpenJDK 8. Floodlight is also built (`~/floodlight/target/floodlight.jar`). Only the Python environment is
missing. `setup.sh` is written for Fedora, so don't use it here.

```bash
cd ~/Documents/Anshul/coding/projects/SDN-routing-multi-class
sudo apt install -y python3.12-venv openvswitch-switch iperf3 ffmpeg openjdk-8-jre-headless
sudo systemctl enable --now openvswitch-switch
sudo modprobe -a openvswitch sch_htb sch_netem

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # if nfstream fails, drop it (only the AIRSDN baseline uses it)

.venv/bin/python -m pytest fattree/tests geant2/tests      # offline checks
.venv/bin/python fattree/fattree_topo.py                   # topology summary
```

## Collecting the dataset

```bash
./start_floodlight.sh                         # terminal 1, keep it running

sudo mn -c                                    # terminal 2: clear leftovers of earlier runs
sudo .venv/bin/python geant2/collect_dataset.py --topology fattree --samples 1000
#   -> fattree/data/fattree_dataset.csv
```

Collection takes about 5.5–6.5 hours for 1,000 rows. Each row is a real-time measurement: a 5 s iperf3
probe, 50 pings and a 2 s queue drain, plus a 5 s warm-up for every new traffic scenario. A GPU cannot
speed this up. Press Ctrl+C at any time and rerun the same command to resume. The CSV has the same columns
as `geant2/data/geant2_dataset.csv`.

`sudo .venv/bin/python fattree/fattree_topo.py --cli` opens the network in the Mininet CLI.
