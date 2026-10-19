

Floodlight must be running in a dedicated terminal before starting the Mininet simulation.

```bash
# Terminal 1: Launch Floodlight Controller
cd /home/rahul/floodlight
java -jar target/floodlight.jar






Execute the main experiment script. It uses trained ML models to evaluate network paths dynamically, selects optimal low-traffic routes via Floodlight REST API, and streams video:

```bash


xhost +SI:localuser:root

cd ~/AI-based-Routing-and-Traffic-Classification-in-SDN/experiment
sudo DISPLAY=:0 python3 ai-based-routing-algorithm-experiment.py

```



Execute the baseline QoS routing experiment to compare performance (RTT, Throughput) against standard shortest-path metric:

```bash
cd /home/rahul/AI-based-Routing-and-Traffic-Classification-in-SDN
sudo python3 experiment/qos-experiment.py
```



Clean up background processes and network interfaces after simulation runs:

```bash
# Clean up temporary video recording file
sudo bash clear_resources/deleteFile.sh records/input.ts

# Reset Mininet network topologies
sudo mn -c
```
