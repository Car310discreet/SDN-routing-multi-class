#!/usr/bin/python

import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from time import sleep, perf_counter
from mininet.log import setLogLevel, info
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch, CPULimitedHost, Host
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.cli import CLI
import json
import pingparsing
import threading
from threading import Thread
import floodlightRestApi
import subprocess
from nfstream import NFStreamer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

net = None
activeThreadList = []
trafficTime = 0

NUM_CANDIDATE_PATHS = 5
PROBE_PRIORITY = 40000      # probe flows (ICMP + TCP only) override the selected path during probing
PROBE_IP_PROTOS = [1, 6]    # ICMP for ping, TCP for iperf3; the UDP video keeps using the selected path
IPERF_PORT = "5555"
# Re-run the routing algorithm every N seconds during video transmission (0 = disabled).
# A probe cycle takes ~4-5 min and saturates the client's access link, so it degrades the
# video it is meant to protect; it is therefore opt-in: --reroute-interval SECONDS
rerouteInterval = 0
selectedPathFlows = []
selectedPathIndex = None

class NsfnetTopo(Topo):
  
    def build( self, **params ):
        # Hosts
        hostCount = 20
        h0 = self.addHost('h0', ip='10.0.0.1', cpu=0.8/hostCount)
        h1 = self.addHost('h1', ip='10.0.0.2', cpu=0.8/hostCount)
        h2 = self.addHost('h2', ip='10.0.0.3', cpu=0.8/hostCount)
        h3 = self.addHost('h3', ip='10.0.0.4', cpu=0.8/hostCount)
        h4 = self.addHost('h4', ip='10.0.0.5', cpu=0.8/hostCount)
        h5 = self.addHost('h5', ip='10.0.0.6', cpu=0.8/hostCount)
        h6 = self.addHost('h6', ip='10.0.0.7', cpu=0.8/hostCount)
        h7 = self.addHost('h7', ip='10.0.0.8', cpu=0.8/hostCount)
        h8 = self.addHost('h8', ip='10.0.0.9', cpu=0.8/hostCount)
        h9 = self.addHost('h9', ip='10.0.0.10', cpu=0.8/hostCount)
        h10 = self.addHost('h10', ip='10.0.0.11', cpu=0.8/hostCount)
        h11 = self.addHost('h11', ip='10.0.0.12', cpu=0.8/hostCount)
        h12 = self.addHost('h12', ip='10.0.0.13', cpu=0.8/hostCount)
        h13 = self.addHost('h13', ip='10.0.0.14', cpu=0.8/hostCount)
        h14 = self.addHost('h14', ip='10.0.0.15', cpu=0.8/hostCount)
        h15 = self.addHost('h15', ip='10.0.0.16', cpu=0.8/hostCount)
        h16 = self.addHost('h16', ip='10.0.0.17', cpu=0.8/hostCount)
        h17 = self.addHost('h17', ip='10.0.0.18', cpu=0.8/hostCount)
        h18 = self.addHost('h18', ip='10.0.0.19', cpu=0.8/hostCount)
        h19 = self.addHost('h19', ip='10.0.0.20', cpu=0.8/hostCount)


        # Switches
        s0 = self.addSwitch('s0', dpid='00:00:00:00:00:00:00:01', protocols="OpenFlow13")
        s1 = self.addSwitch('s1', dpid='00:00:00:00:00:00:00:02', protocols="OpenFlow13")
        s2 = self.addSwitch('s2', dpid='00:00:00:00:00:00:00:03', protocols="OpenFlow13")
        s3 = self.addSwitch('s3', dpid='00:00:00:00:00:00:00:04', protocols="OpenFlow13")
        s4 = self.addSwitch('s4', dpid='00:00:00:00:00:00:00:05', protocols="OpenFlow13")
        s5 = self.addSwitch('s5', dpid='00:00:00:00:00:00:00:06', protocols="OpenFlow13")
        s6 = self.addSwitch('s6', dpid='00:00:00:00:00:00:00:07', protocols="OpenFlow13")
        s7 = self.addSwitch('s7', dpid='00:00:00:00:00:00:00:08', protocols="OpenFlow13")
        s8 = self.addSwitch('s8', dpid='00:00:00:00:00:00:00:09', protocols="OpenFlow13")
        s9 = self.addSwitch('s9', dpid='00:00:00:00:00:00:00:10', protocols="OpenFlow13")
        s10 = self.addSwitch('s10', dpid='00:00:00:00:00:00:00:11', protocols="OpenFlow13")
        s11 = self.addSwitch('s11', dpid='00:00:00:00:00:00:00:12', protocols="OpenFlow13")
        s12 = self.addSwitch('s12', dpid='00:00:00:00:00:00:00:13', protocols="OpenFlow13")
        s13 = self.addSwitch('s13', dpid='00:00:00:00:00:00:00:14', protocols="OpenFlow13")

        # Links
        # Link options
        linkopts1 = dict(delay='25ms', bw=10, loss=0, max_queue_size=1000, use_htb=True, cls=TCLink) # between switch and host
        linkopts2 = dict(delay='25ms', bw=10, loss=0, max_queue_size=1000, use_htb=True, cls=TCLink) # between 2 switches

        # Switch to switch
        self.addLink(s0, s1, **linkopts2)
        self.addLink(s0, s2, **linkopts2)
        self.addLink(s0, s3, **linkopts2)
        self.addLink(s1, s7, **linkopts2)
        self.addLink(s1, s2, **linkopts2)
        self.addLink(s2, s5, **linkopts2)
        self.addLink(s3, s4, **linkopts2)
        self.addLink(s3, s8, **linkopts2)
        self.addLink(s4, s5, **linkopts2)
        self.addLink(s4, s6, **linkopts2)
        self.addLink(s5, s12, **linkopts2)
        self.addLink(s5, s13, **linkopts2)
        self.addLink(s6, s7, **linkopts2)
        self.addLink(s7, s10, **linkopts2)
        self.addLink(s8, s9, **linkopts2)
        self.addLink(s8, s11, **linkopts2)
        self.addLink(s9, s10, **linkopts2)
        self.addLink(s9, s12, **linkopts2)
        self.addLink(s10, s11, **linkopts2)
        self.addLink(s10, s13, **linkopts2)
        self.addLink(s11, s12, **linkopts2)

        # switch to host
        self.addLink(s0, h0, **linkopts1)
        self.addLink(s1, h1, **linkopts1)
        self.addLink(s2, h2, **linkopts1)
        self.addLink(s3, h3, **linkopts1)
        self.addLink(s4, h4, **linkopts1)
        self.addLink(s5, h5, **linkopts1)
        self.addLink(s7, h6, **linkopts1)
        self.addLink(s8, h7, **linkopts1)
        self.addLink(s9, h8, **linkopts1)
        self.addLink(s10, h9, **linkopts1)
        self.addLink(s11, h10, **linkopts1)

        #-------------------------------------
        self.addLink(s3, h11, **linkopts1)
        self.addLink(s1, h12, **linkopts1)
        self.addLink(s8, h13, **linkopts1)
        self.addLink(s7, h14, **linkopts1)
        self.addLink(s11, h15, **linkopts1)
        self.addLink(s2, h16, **linkopts1)
        self.addLink(s10, h17, **linkopts1)
        self.addLink(s5, h18, **linkopts1)
        self.addLink(s3, h19, **linkopts1)
        
def startNetwork():
    start = perf_counter() # time.perf_counter()

    global net
    global activeThreadList
    global trafficTime
    trafficTime = 900 # must outlast path probing (~5 min), measurement and video so background traffic stays present
    activeThreadList = []
    net = None

    net = Mininet(topo=NsfnetTopo(), link=TCLink, build=False, switch=OVSKernelSwitch, autoSetMacs=True, waitConnected=True)

    remote_ip = "127.0.0.1"
    info('[INFO]****** Adding Floodlight Controller *****\n')
    net.addController('c1', controller=RemoteController, host=CPULimitedHost, ip=remote_ip, port=6653, protocols="OpenFlow13")

    # Build the network
    info('[INFO]****** Building Network *****\n')
    net.build() 
    info('[INFO]****** Starting Network *****\n')
    net.start()
    sleep(7) 

    # Configure controller
    info('[INFO]****** Configuring Controller *****\n')
    floodlightRestApi.deleteAllFlows() # static flows persist in Floodlight across runs
    floodlightRestApi.setRoutingMetric("hopcount")
    # floodlightRestApi.enableSwitchStats()

    info('[INFO]****** Testing Connectivity Between Hosts *****\n')
    net.pingAll()


    info('[INFO]****** Setting Paths Among Hosts to Generate Traffic *****\n')
    setTrafficPaths()

    # Start traffic
    info(f'[INFO]****** Active thread count: {threading.active_count()}\n')
    info('[INFO]****** Generating artificial traffic *****\n')
    time_1 = perf_counter()
    print("Time Passed: ", time_1 - start)

    generateTraffic(sender="h7", receiver="h3", getStats=False, bandWidth="100M")
    generateTraffic(sender="h13", receiver="h11", getStats=False, bandWidth="100M")

    generateTraffic(sender="h6", receiver="h1", getStats=False, bandWidth="100M")
    generateTraffic(sender="h14", receiver="h12", getStats=False, bandWidth="100M")

    generateTraffic(sender="h9", receiver="h10", getStats=False, bandWidth="100M")
    generateTraffic(sender="h17", receiver="h15", getStats=False, bandWidth="100M")

    generateTraffic(sender="h5", receiver="h2", getStats=False, bandWidth="100M")
    generateTraffic(sender="h18", receiver="h16", getStats=False, bandWidth="100M")

    generateTraffic(sender="h4", receiver="h19", getStats=False, bandWidth="3M")


    sleep(13) # time.sleep
    info(f'[INFO]****** Active thread count: {threading.active_count()}\n')

    routingAlgorithm(src_ipv4="10.0.0.1", dst_ipv4="10.0.0.9")

    # Same 60 s RTT / throughput measurement as the Floodlight baselines (paper Figs. 16-17)
    info(f'[INFO]****** Statistics Collecting *****\n')
    getPingStats("h0", "h8", getStats=True)
    getThroughput(sender = "h8", receiver = "h0", bandWidth="0")

    # Start Video Stream
    info(f'[INFO]****** Video Stream Starting *****\n')
    startStream(sender = "h8", receiver = "h0")
    info(f'[INFO]****** Video Stream Ended *****\n')
    info(f'[INFO]****** Active thread count: {threading.active_count()}\n')

    # Evaluate Video Quality (PSNR & SSIM)
    info(f'[INFO]****** Evaluating Video Path Quality (PSNR & SSIM) *****\n')
    try:
        psnr_val = calculatePSNR()
        info(f'[INFO]****** PSNR Result: {psnr_val} dB *****\n')
    except Exception as exc:
        info(f'[WARN]****** PSNR Evaluation failed: {exc} *****\n')

    try:
        ssim_first, ssim_second = calculateSSIM()
        info(f'[INFO]****** SSIM Result: All={ssim_first} ({ssim_second}) *****\n')
    except Exception as exc:
        info(f'[WARN]****** SSIM Evaluation failed: {exc} *****\n')

    time_2 = perf_counter()
    print("Time passed: ", time_2 - start)

    
 

    info(f'[INFO]****** Active thread count: {threading.active_count()}\n')

   

    info(f'[INFO]****** Active thread count: {threading.active_count()}\n')

    for thread in activeThreadList:
        print(thread)
        print("Thread joining...")
        thread.join(timeout=5)
        print("Thread done...")

    info(f'[INFO]****** Mininet Cleaning *****\n')
    cleanMininet()
    sleep(5)

    end = perf_counter() # time.perf_counter()
    diff = end - start
    info(f'[INFO]****** Completed in {diff} seconds *****\n')
    # sys.exit()

def routingAlgorithm(src_ipv4, dst_ipv4, stopEvent=None):
    """
        AI-based routing algorithm (paper Table 2). Probes each candidate path with
        ping + iperf3 + NFStream, predicts its traffic level and installs the path
        with the lowest predicted level. Returns the selected path index, or None if
        stopEvent was set before all candidates were probed.
    """
    model_path = os.path.join(get_project_dir(), 'models', 'logistic_regression_model.pkl')
    loaded_model = joblib.load(model_path)
    paths = []

    src_switch_dpid, _ = floodlightRestApi.getSwitchAndPortByHost(src_ipv4)
    dst_switch_dpid, _ = floodlightRestApi.getSwitchAndPortByHost(dst_ipv4)

    for i in range(NUM_CANDIDATE_PATHS):
        if stopEvent is not None and stopEvent.is_set():
            info('[INFO]****** Routing cycle aborted *****\n')
            return None
        flows = floodlightRestApi.pathPusher(src_ipv4, dst_ipv4, num_paths=NUM_CANDIDATE_PATHS, path_index=i,
                                             name_prefix="probe", priority=PROBE_PRIORITY, ip_protos=PROBE_IP_PROTOS)
        info(f'[INFO]****** Testing The Path {i} Performance *****\n')
        path_statistics = []
    
        info(f'[INFO]****** PATH = {i} Ping Stats *****\n')
        avgRTT, packetLoss = getPingStats("h0", "h8")
        path_statistics.append(avgRTT)
        path_statistics.append(packetLoss)
        print(f"""*** Average RTT: {avgRTT}\nPacket Loss: {packetLoss}""")

        # Iperf tcp stats
        info(f'[INFO]****** Path {i} Iperf TCP Stats *****\n')
        iperfResult = generateTraffic(sender = "h8", receiver = "h0", getStats = True, bandWidth="0")
        tcpFlowResult = iperfResult[0]

        try:
            sum_received = tcpFlowResult.get("sum_received", {}) or tcpFlowResult.get("sum", {})
            sum_sent = tcpFlowResult.get("sum_sent", {}) or tcpFlowResult.get("sum", {})
            bits_per_sec = float(sum_received.get("bits_per_second", 0.0))
            retransmits = int(sum_sent.get("retransmits", 0))
            cpu_util = tcpFlowResult.get("cpu_utilization_percent", {})
            cpu_host_total = float(cpu_util.get("host_total", 0.0))
            cpu_host_system = float(cpu_util.get("host_system", 0.0))
        except (KeyError, TypeError, ValueError):
            bits_per_sec = 0.0
            retransmits = 0
            cpu_host_total = 0.0
            cpu_host_system = 0.0

        path_statistics.append(bits_per_sec)
        path_statistics.append(100 - ((bits_per_sec / 10000000) * 100))
        path_statistics.append(retransmits)
        path_statistics.append(cpu_host_total)
        path_statistics.append(cpu_host_system)

        nfstreamResult = iperfResult[1]
        if len(nfstreamResult) > 0:
            flow_data = nfstreamResult[0]
            path_statistics.append(flow_data.bidirectional_duration_ms)
            path_statistics.append(flow_data.src2dst_duration_ms)
            path_statistics.append(flow_data.dst2src_duration_ms)
            path_statistics.append(flow_data.bidirectional_mean_piat_ms)
            path_statistics.append(flow_data.bidirectional_stddev_piat_ms)
            path_statistics.append(flow_data.bidirectional_max_piat_ms)
            path_statistics.append(flow_data.src2dst_mean_piat_ms)
            path_statistics.append(flow_data.src2dst_stddev_piat_ms)
            path_statistics.append(flow_data.src2dst_max_piat_ms)
            path_statistics.append(flow_data.dst2src_mean_piat_ms)
            path_statistics.append(flow_data.dst2src_stddev_piat_ms)
            path_statistics.append(flow_data.dst2src_max_piat_ms)
        else:
            path_statistics.extend([0] * 12)

        path_statistics = np.array([path_statistics])
        prediction = loaded_model.predict(path_statistics)
        prediction_value = float(prediction[0]) if hasattr(prediction, '__len__') and len(prediction) > 0 else float(prediction)
        print(f"[INFO] Path {i} prediction ----> {prediction_value}")

        candidate_path, candidate_latency, candidate_hop_count = floodlightRestApi.getPathById(src_switch_dpid, dst_switch_dpid, NUM_CANDIDATE_PATHS, i)
        print(f"[INFO] Candidate path {i} details -> latency={candidate_latency} ms, hop_count={candidate_hop_count}, path={candidate_path}")

        paths.append({
            'pathIndex': i,
            'pathTrafficValue': prediction_value,
            'latency': candidate_latency,
            'hopCount': candidate_hop_count,
            'path': candidate_path,
        })

        for flow in flows:
            floodlightRestApi.deleteFlowByName(flow)
        sleep(1)

    print(f"[INFO] Calculating best path based on traffic value from {len(paths)} candidate paths...")
    best_path = paths[0]
    for candidate in paths[1:]:
        if candidate['pathTrafficValue'] < best_path['pathTrafficValue']:
            best_path = candidate

    print(f"[INFO] Best path selected: index={best_path['pathIndex']}, traffic_score={best_path['pathTrafficValue']}, latency={best_path['latency']} ms, hop_count={best_path['hopCount']}, path={best_path['path']}")
    setSelectedPath(src_ipv4, dst_ipv4, best_path['pathIndex'])
    return best_path['pathIndex']

def setSelectedPath(src_ipv4, dst_ipv4, path_index):
    """Install the selected path for all IPv4 traffic between the hosts, replacing the previous one."""
    global selectedPathFlows, selectedPathIndex
    if path_index == selectedPathIndex:
        print(f"[INFO] Keeping current path {path_index} for traffic between {src_ipv4} and {dst_ipv4}")
        return
    print(f"[INFO] Setting best path {path_index} for traffic between {src_ipv4} and {dst_ipv4}")
    # Push the new path before removing stale entries so traffic is never left without a route
    flows = floodlightRestApi.pathPusher(src_ipv4, dst_ipv4, num_paths=NUM_CANDIDATE_PATHS, path_index=path_index, name_prefix="best")
    for flow in selectedPathFlows:
        if flow not in flows:
            floodlightRestApi.deleteFlowByName(flow)
    selectedPathFlows = flows
    selectedPathIndex = path_index

class RerouteMonitor(Thread):
    """Periodically re-executes the routing algorithm while the video is being transmitted."""
    def __init__(self, src_ipv4: str, dst_ipv4: str, interval: float):
        Thread.__init__(self, daemon=True)
        self.src_ipv4 = src_ipv4
        self.dst_ipv4 = dst_ipv4
        self.interval = interval
        self.stopEvent = threading.Event()

    def run(self):
        while not self.stopEvent.wait(self.interval):
            info('[INFO]****** Periodic re-routing: re-evaluating candidate paths *****\n')
            routingAlgorithm(self.src_ipv4, self.dst_ipv4, stopEvent=self.stopEvent)

    def stop(self):
        self.stopEvent.set()


def calculatePSNR():
    project_dir = get_project_dir()
    videoSource = os.path.join(project_dir, "assets", "surreal.ts")
    outputSource = os.path.join(project_dir, "records", "input.ts")
    if not os.path.exists(outputSource):
        info(f"[WARN] Output video file {outputSource} not found for PSNR calculation.\n")
        return 0.0
    # Offline file comparison: run in the root namespace, Mininet host shells may be busy with traffic
    command = ["ffmpeg", "-hide_banner", "-i", videoSource, "-i", outputSource, "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"]
    psnrResult = subprocess.run(command, capture_output=True, text=True).stderr
    target_line = ""
    for line in psnrResult.split("\n"):
        if "average:" in line:
            target_line = line
            break
    if not target_line:
        lines = [l for l in psnrResult.split("\n") if l.strip()]
        target_line = lines[-2] if len(lines) >= 2 else (lines[0] if lines else "")
    try:
        psnrVal = float(target_line.split("average:")[1].split(" ")[0].strip())
    except Exception as exc:
        info(f"[WARN] PSNR calculation parsing failed: {exc}\n")
        psnrVal = 0.0
    return psnrVal

def calculateSSIM():
    project_dir = get_project_dir()
    videoSource = os.path.join(project_dir, "assets", "surreal.ts")
    outputSource = os.path.join(project_dir, "records", "input.ts")
    if not os.path.exists(outputSource):
        info(f"[WARN] Output video file {outputSource} not found for SSIM calculation.\n")
        return 0.0, 0.0
    # Offline file comparison: run in the root namespace, Mininet host shells may be busy with traffic
    command = ["ffmpeg", "-hide_banner", "-i", videoSource, "-i", outputSource, "-lavfi", "[0:v][1:v]ssim", "-f", "null", "-"]
    ssimResult = subprocess.run(command, capture_output=True, text=True).stderr
    target_line = ""
    for line in ssimResult.split("\n"):
        if "All:" in line:
            target_line = line
            break
    if not target_line:
        lines = [l for l in ssimResult.split("\n") if l.strip()]
        target_line = lines[-2] if len(lines) >= 2 else (lines[0] if lines else "")
    try:
        firstReulst = float(target_line.split("All:")[1].split(" ")[0])
        second_str = target_line.split("All:")[1].split(" ")[1].replace("(", "").replace(")", "").strip()
        secondResult = float(second_str)
    except Exception as exc:
        info(f"[WARN] SSIM calculation parsing failed: {exc}\n")
        firstReulst, secondResult = 0.0, 0.0
    return firstReulst, secondResult

def get_project_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def deleteFile(fileName: str):
    if os.path.exists(fileName):
        os.remove(fileName)
    else:
        script_path = os.path.join(get_project_dir(), 'clear_resources', 'deleteFile.sh')
        if os.path.exists(script_path):
            subprocess.run(['bash', script_path, fileName])

def cleanMininet():
    global net
    if net is None:
        return
    stoppingNet, net = net, None
    # Background iperf3 commands outlive the experiment; interrupt them, net.stop() needs the host shells
    for host in stoppingNet.hosts:
        if host.waiting:
            host.sendInt()
    for thread in activeThreadList:
        thread.join(timeout=10)
    stoppingNet.stop()

def startVideoPreview(videoPath: str):
    if not os.path.exists(videoPath):
        info(f'[WARN]****** Video preview file not found: {videoPath} *****\n')
        return None

    previewEnvironment = os.environ.copy()
    previewEnvironment.setdefault('DISPLAY', ':0')

    try:
        process = subprocess.Popen(
            ['ffplay', '-hide_banner', '-loglevel', 'error', '-autoexit',
             '-window_title', 'Video Streaming', videoPath],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=previewEnvironment,
        )
        sleep(1)
        if process.poll() is not None:
            err = (process.stderr.read() if process.stderr else '').strip()
            info(f'[WARN]****** ffplay exited immediately: {err or "no output"} *****\n')
            return None
        return process
    except Exception as exc:
        info(f'[WARN]****** Failed to open video preview window: {exc} *****\n')
        return None

def startStream(sender:str, receiver:str):
    senderNode, receiverNode = net.getNodeByName(sender), net.getNodeByName(receiver)
    port = "1234"
    project_dir = get_project_dir()
    records_dir = os.path.join(project_dir, "records")
    os.makedirs(records_dir, exist_ok=True)

    videoSource = os.path.join(project_dir, "assets", "surreal.ts")
    outputSource = os.path.join(records_dir, "input.ts")
    receiverUrl = f"udp://{receiverNode.IP()}:{port}"

    senderCommand = ["ffmpeg", "-re", "-i", videoSource, "-c", "copy", "-f", "mpegts", receiverUrl]
    receiverCommand = ["ffmpeg", "-y", "-i", receiverUrl, "-c", "copy", outputSource]

    previewProcess = startVideoPreview(videoSource)

    # popen (not Node.cmd) keeps the hosts' shells free, so the re-route monitor can probe h0/h8
    with open("receiver.log", "w") as receiverLog, open("sender.log", "w") as senderLog:
        receiverProcess = receiverNode.popen(receiverCommand, stdout=receiverLog, stderr=subprocess.STDOUT)
        sleep(0.5)
        senderProcess = senderNode.popen(senderCommand, stdout=senderLog, stderr=subprocess.STDOUT)

        monitor = None
        if rerouteInterval > 0:
            monitor = RerouteMonitor(receiverNode.IP(), senderNode.IP(), rerouteInterval)
            monitor.start()

        senderProcess.wait()
        if monitor is not None:
            monitor.stop()
        sleep(2)
        info(f'[INFO]****** Stopping video receiver *****\n')
        receiverProcess.terminate() # ffmpeg finalizes the output file on SIGTERM
        try:
            receiverProcess.wait(timeout=10)
        except subprocess.TimeoutExpired:
            receiverProcess.kill()
        if previewProcess is not None:
            previewProcess.terminate()
        if monitor is not None:
            monitor.join()

def getPingStats(sender: str, receiver: str, getStats = False) -> tuple[float, float]:
    senderNode, receiverNode = net.getNodeByName(sender), net.getNodeByName(receiver)
    receiverIpv4 = receiverNode.IP()
    print("Receiver ipv4 -> ", receiverIpv4)

    if getStats == True:
        packetCount = "60"
        command = f"ping {receiverIpv4} -c {packetCount} > output_ai_ping.txt"
    else:
        packetCount = "500"
        command = f"ping {receiverIpv4} -c {packetCount} -f"

    result = senderNode.cmd(command)

    ping_parser = pingparsing.PingParsing()
    result = ping_parser.parse(result).as_dict()
    packetLoss = result["packet_loss_rate"]
    avgRTT = result["rtt_avg"]
    return avgRTT, packetLoss

def generateTraffic(sender: str, receiver: str, getStats: bool, bandWidth: str):
    global activeThreadList
    global trafficTime
    server, client = net.getNodeByName(sender), net.getNodeByName(receiver)
    port = "5555"
    if getStats:
        time = "30"
    else:
        time = str(trafficTime)

    serverCommand = f"iperf3 -s -p {port} -i 1 -1"
    clientCommand = f"iperf3 -c {server.IP()} -p {port} -b {bandWidth} -R -t {time} -J"

    serverThread = HostCommand(server, serverCommand)
    serverThread.daemon = True
    clientThread = HostCommand(client, clientCommand) 
    clientThread.daemon = True
    if getStats == False:
        activeThreadList.append(serverThread)
        activeThreadList.append(clientThread)

    streamListener = StreamListener("s0-eth4")
    streamListener.daemon = True
    if getStats:
        streamListener.start()
    sleep(1)
    serverThread.start()
    sleep(1) # time.sleep
    clientThread.start()

    
    if getStats: 
        serverThread.join()
        clientThread.join()
        streamListener.flag = False
        client.cmd(f"ping {server.IP()} -c 5 -f")
        streamListener.join()
        try:
            result = json.loads(clientThread.result)["end"]
        except Exception:
            result = {}
        return result, streamListener.stream
    
def getThroughput(sender: str, receiver: str, bandWidth: str):
    server, client = net.getNodeByName(sender), net.getNodeByName(receiver)
    port = "5555"
    time = "60"

    serverCommand = f"iperf3 -s -p {port} -i 1 -1"
    clientCommand = f"iperf3 -c {server.IP()} -p {port} -b {bandWidth} -R -t {time} > output_ai_throughput.txt"

    serverThread = HostCommand(server, serverCommand)
    serverThread.daemon = True
    clientThread = HostCommand(client, clientCommand) 
    clientThread.daemon = True

    sleep(1)
    serverThread.start()
    sleep(1) # time.sleep
    clientThread.start()

    serverThread.join()
    clientThread.join()

class HostCommand(Thread):
    def __init__(self, host: Host, command: str):
        Thread.__init__(self)
        self._host = host
        self._command = command
        self.result = None
    def run(self):
        self.result = self._host.cmd(self._command)

class StreamListener(Thread):
    def __init__(self, interface: str):
        Thread.__init__(self)
        self.interface = interface
        self.stream = []
        self.flag = True

    def run(self):
        flow_streamer = NFStreamer(source=self.interface,
                        statistical_analysis=True,
                        idle_timeout=15)
        self.stream = []
        for flow in flow_streamer:
            # The iperf3 TCP connection between client and server (the same flow the training data was
            # collected from); never the UDP video stream or ICMP probes
            if (flow.src_ip in ("10.0.0.1", "10.0.0.9") and flow.dst_ip in ("10.0.0.1", "10.0.0.9")
                    and flow.protocol == 6 and int(IPERF_PORT) in (flow.src_port, flow.dst_port)):
                self.stream.append(flow)
                print("[INFO] NFStream captured flow: ", flow.src_ip, " -> ", flow.dst_ip)
                break
            if(self.flag == False):
                break

def setTrafficPaths():
    # H7 --> H3
    flow_h3_h7_1 = {
        "switch": "00:00:00:00:00:00:00:04",
        "name": "flow_h3_h7_1",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.4/32",
        "ipv4_dst": "10.0.0.8/32",
        "in_port":"4",
        "active":"true",
        "actions":"output=3"
    }
    flow_h3_h7_1_r = {
        "switch": "00:00:00:00:00:00:00:04",
        "name": "flow_h3_h7_1_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.8/32",
        "ipv4_dst": "10.0.0.4/32",
        "in_port":"3",
        "active":"true",
        "actions":"output=4"
    }
    floodlightRestApi.flowPusher(flow_h3_h7_1)
    floodlightRestApi.flowPusher(flow_h3_h7_1_r)
    #-------------------------------
    flow_h3_h7_2 = {
        "switch": "00:00:00:00:00:00:00:09",
        "name": "flow_h3_h7_2",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.4/32",
        "ipv4_dst": "10.0.0.8/32",
        "in_port":"1",
        "active":"true",
        "actions":"output=4"
    }
    flow_h3_h7_2_r = {
        "switch": "00:00:00:00:00:00:00:09",
        "name": "flow_h3_h7_2_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.8/32",
        "ipv4_dst": "10.0.0.4/32",
        "in_port":"4",
        "active":"true",
        "actions":"output=1"
    }
    floodlightRestApi.flowPusher(flow_h3_h7_2)
    floodlightRestApi.flowPusher(flow_h3_h7_2_r)
    #-------------------------------
    #-------------------------------
    # H11 --> H13
    flow_h11_h13_1 = {
        "switch": "00:00:00:00:00:00:00:04",
        "name": "flow_h11_h13_1",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.12/32",
        "ipv4_dst": "10.0.0.14/32",
        "in_port":"5",
        "active":"true",
        "actions":"output=3"
    }
    flow_h11_h13_1_r = {
        "switch": "00:00:00:00:00:00:00:04",
        "name": "flow_h11_h13_1_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.14/32",
        "ipv4_dst": "10.0.0.12/32",
        "in_port":"3",
        "active":"true",
        "actions":"output=5"
    }    
    floodlightRestApi.flowPusher(flow_h11_h13_1)
    floodlightRestApi.flowPusher(flow_h11_h13_1_r)
    #-------------------------------
    flow_h11_h13_2 = {
        "switch": "00:00:00:00:00:00:00:09",
        "name": "flow_h11_h13_2",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.12/32",
        "ipv4_dst": "10.0.0.14/32",
        "in_port":"1",
        "active":"true",
        "actions":"output=5"
    }
    flow_h11_h13_2_r = {
        "switch": "00:00:00:00:00:00:00:09",
        "name": "flow_h11_h13_2_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.14/32",
        "ipv4_dst": "10.0.0.12/32",
        "in_port":"5",
        "active":"true",
        "actions":"output=1"
    }
    floodlightRestApi.flowPusher(flow_h11_h13_2)
    floodlightRestApi.flowPusher(flow_h11_h13_2_r)
    #-------------------------------
    #-------------------------------
    # H1 --> H6
    flow_h1_h6_1 = {
        "switch": "00:00:00:00:00:00:00:02",
        "name": "flow_h1_h6_1",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.2/32",
        "ipv4_dst": "10.0.0.7/32",
        "in_port":"4",
        "active":"true",
        "actions":"output=2"
    }
    flow_h1_h6_1_r = {
        "switch": "00:00:00:00:00:00:00:02",
        "name": "flow_h1_h6_1_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.7/32",
        "ipv4_dst": "10.0.0.2/32",
        "in_port":"2",
        "active":"true",
        "actions":"output=4"
    }
    floodlightRestApi.flowPusher(flow_h1_h6_1)
    floodlightRestApi.flowPusher(flow_h1_h6_1_r)
    #-------------------------------
    flow_h1_h6_2 = {
        "switch": "00:00:00:00:00:00:00:08",
        "name": "flow_h1_h6_2",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.2/32",
        "ipv4_dst": "10.0.0.7/32",
        "in_port":"1",
        "active":"true",
        "actions":"output=4"
    }
    flow_h1_h6_2_r = {
        "switch": "00:00:00:00:00:00:00:08",
        "name": "flow_h1_h6_2_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.7/32",
        "ipv4_dst": "10.0.0.2/32",
        "in_port":"4",
        "active":"true",
        "actions":"output=1"
    }
    floodlightRestApi.flowPusher(flow_h1_h6_2)
    floodlightRestApi.flowPusher(flow_h1_h6_2_r)
    #-------------------------------
    #-------------------------------
    # H14 --> H12
    flow_h12_h14_1 = {
        "switch": "00:00:00:00:00:00:00:02",
        "name": "flow_h12_h14_1",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.13/32",
        "ipv4_dst": "10.0.0.15/32",
        "in_port":"5",
        "active":"true",
        "actions":"output=2"
    }
    flow_h12_h14_1_r = {
        "switch": "00:00:00:00:00:00:00:02",
        "name": "flow_h12_h14_1_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.15/32",
        "ipv4_dst": "10.0.0.13/32",
        "in_port":"2",
        "active":"true",
        "actions":"output=5"
    }
    floodlightRestApi.flowPusher(flow_h12_h14_1)
    floodlightRestApi.flowPusher(flow_h12_h14_1_r)
    #-------------------------------
    flow_h12_h14_2 = {
        "switch": "00:00:00:00:00:00:00:08",
        "name": "flow_h12_h14_2",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.13/32",
        "ipv4_dst": "10.0.0.15/32",
        "in_port":"1",
        "active":"true",
        "actions":"output=5"
    }
    flow_h12_h14_2_r = {
        "switch": "00:00:00:00:00:00:00:08",
        "name": "flow_h12_h14_2_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.15/32",
        "ipv4_dst": "10.0.0.13/32",
        "in_port":"5",
        "active":"true",
        "actions":"output=1"
    }
    floodlightRestApi.flowPusher(flow_h12_h14_2)
    floodlightRestApi.flowPusher(flow_h12_h14_2_r)
    #-------------------------------
    #-------------------------------
    # H10 --> H9
    flow_h10_h9_1 = {
        "switch": "00:00:00:00:00:00:00:12",
        "name": "flow_h10_h9_1",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.11/32",
        "ipv4_dst": "10.0.0.10/32",
        "in_port":"4",
        "active":"true",
        "actions":"output=2"
    }
    flow_h10_h9_1_r = {
        "switch": "00:00:00:00:00:00:00:12",
        "name": "flow_h10_h9_1_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.10/32",
        "ipv4_dst": "10.0.0.11/32",
        "in_port":"2",
        "active":"true",
        "actions":"output=4"
    }
    floodlightRestApi.flowPusher(flow_h10_h9_1)
    floodlightRestApi.flowPusher(flow_h10_h9_1_r)
    #-------------------------------
    flow_h10_h9_2 = {
        "switch": "00:00:00:00:00:00:00:11",
        "name": "flow_h10_h9_2",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.11/32",
        "ipv4_dst": "10.0.0.10/32",
        "in_port":"3",
        "active":"true",
        "actions":"output=5"
    }
    flow_h10_h9_2_r = {
        "switch": "00:00:00:00:00:00:00:11",
        "name": "flow_h10_h9_2_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.10/32",
        "ipv4_dst": "10.0.0.11/32",
        "in_port":"5",
        "active":"true",
        "actions":"output=3"
    }
    floodlightRestApi.flowPusher(flow_h10_h9_2)
    floodlightRestApi.flowPusher(flow_h10_h9_2_r)
    #-------------------------------
    #-------------------------------
    # H15 --> H17
    flow_h15_h17_1 = {
        "switch": "00:00:00:00:00:00:00:12",
        "name": "flow_h15_h17_1",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.16/32",
        "ipv4_dst": "10.0.0.18/32",
        "in_port":"5",
        "active":"true",
        "actions":"output=2"
    }
    flow_h15_h17_1_r = {
        "switch": "00:00:00:00:00:00:00:12",
        "name": "flow_h15_h17_1_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.18/32",
        "ipv4_dst": "10.0.0.16/32",
        "in_port":"2",
        "active":"true",
        "actions":"output=5"
    }
    floodlightRestApi.flowPusher(flow_h15_h17_1)
    floodlightRestApi.flowPusher(flow_h15_h17_1_r)
    #-------------------------------
    flow_h15_h17_2 = {
        "switch": "00:00:00:00:00:00:00:11",
        "name": "flow_h15_h17_2",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.16/32",
        "ipv4_dst": "10.0.0.18/32",
        "in_port":"3",
        "active":"true",
        "actions":"output=6"
    }
    flow_h15_h17_2_r = {
        "switch": "00:00:00:00:00:00:00:11",
        "name": "flow_h15_h17_2_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.18/32",
        "ipv4_dst": "10.0.0.16/32",
        "in_port":"6",
        "active":"true",
        "actions":"output=3"
    }
    floodlightRestApi.flowPusher(flow_h15_h17_2)
    floodlightRestApi.flowPusher(flow_h15_h17_2_r)
    #-------------------------------
    #-------------------------------
    # H2 --> H5
    flow_h2_h5_1 = {
        "switch": "00:00:00:00:00:00:00:03",
        "name": "flow_h2_h5_1",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.3/32",
        "ipv4_dst": "10.0.0.6/32",
        "in_port":"4",
        "active":"true",
        "actions":"output=3"
    }
    flow_h2_h5_1_r = {
        "switch": "00:00:00:00:00:00:00:03",
        "name": "flow_h2_h5_1_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.6/32",
        "ipv4_dst": "10.0.0.3/32",
        "in_port":"3",
        "active":"true",
        "actions":"output=4"
    }
    floodlightRestApi.flowPusher(flow_h2_h5_1)
    floodlightRestApi.flowPusher(flow_h2_h5_1_r)
    #-------------------------------
    flow_h2_h5_2 = {
        "switch": "00:00:00:00:00:00:00:06",
        "name": "flow_h2_h5_2",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.3/32",
        "ipv4_dst": "10.0.0.6/32",
        "in_port":"1",
        "active":"true",
        "actions":"output=5"
    }
    flow_h2_h5_2_r = {
        "switch": "00:00:00:00:00:00:00:06",
        "name": "flow_h2_h5_2_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.6/32",
        "ipv4_dst": "10.0.0.3/32",
        "in_port":"5",
        "active":"true",
        "actions":"output=1"
    }
    floodlightRestApi.flowPusher(flow_h2_h5_2)
    floodlightRestApi.flowPusher(flow_h2_h5_2_r)
    #-------------------------------
    #-------------------------------
    # H16 --> H18
    flow_h16_h18_1 = {
        "switch": "00:00:00:00:00:00:00:03",
        "name": "flow_h16_h18_1",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.17/32",
        "ipv4_dst": "10.0.0.19/32",
        "in_port":"5",
        "active":"true",
        "actions":"output=3"
    }
    flow_h16_h18_1_r = {
        "switch": "00:00:00:00:00:00:00:03",
        "name": "flow_h16_h18_1_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.19/32",
        "ipv4_dst": "10.0.0.17/32",
        "in_port":"3",
        "active":"true",
        "actions":"output=5"
    }
    floodlightRestApi.flowPusher(flow_h16_h18_1)
    floodlightRestApi.flowPusher(flow_h16_h18_1_r)
    #-------------------------------
    flow_h16_h18_2 = {
        "switch": "00:00:00:00:00:00:00:06",
        "name": "flow_h16_h18_2",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.17/32",
        "ipv4_dst": "10.0.0.19/32",
        "in_port":"1",
        "active":"true",
        "actions":"output=6"
    }
    flow_h16_h18_2_r = {
        "switch": "00:00:00:00:00:00:00:06",
        "name": "flow_h16_h18_2_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.19/32",
        "ipv4_dst": "10.0.0.17/32",
        "in_port":"6",
        "active":"true",
        "actions":"output=1"
    }
    floodlightRestApi.flowPusher(flow_h16_h18_2)
    floodlightRestApi.flowPusher(flow_h16_h18_2_r)
    #-------------------------------
    #-------------------------------
    # H19 --> H4
    flow_h19_h4_1 = {
        "switch": "00:00:00:00:00:00:00:04",
        "name": "flow_h19_h4_1",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.20/32",
        "ipv4_dst": "10.0.0.5/32",
        "in_port":"6",
        "active":"true",
        "actions":"output=2"
    }
    flow_h19_h4_1_r = {
        "switch": "00:00:00:00:00:00:00:04",
        "name": "flow_h19_h4_1_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.5/32",
        "ipv4_dst": "10.0.0.20/32",
        "in_port":"2",
        "active":"true",
        "actions":"output=6"
    }
    floodlightRestApi.flowPusher(flow_h19_h4_1)
    floodlightRestApi.flowPusher(flow_h19_h4_1_r)
    #-------------------------------
    flow_h19_h4_2 = {
        "switch": "00:00:00:00:00:00:00:05",
        "name": "flow_h19_h4_2",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.20/32",
        "ipv4_dst": "10.0.0.5/32",
        "in_port":"1",
        "active":"true",
        "actions":"output=4"
    }
    flow_h19_h4_2_r = {
        "switch": "00:00:00:00:00:00:00:05",
        "name": "flow_h19_h4_2_r",
        "cookie": "0",
        "priority":"32768",
        "eth_type" : "0x0800" ,
        "ipv4_src": "10.0.0.5/32",
        "ipv4_dst": "10.0.0.20/32",
        "in_port":"4",
        "active":"true",
        "actions":"output=1"
    }
    floodlightRestApi.flowPusher(flow_h19_h4_2)
    floodlightRestApi.flowPusher(flow_h19_h4_2_r)
    #-------------------------------
    #-------------------------------
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="AI-based routing experiment (AIRSDN)")
    parser.add_argument("--reroute-interval", type=float, default=0,
                        help="re-run the routing algorithm every N seconds during video transmission (0 = disabled)")
    args = parser.parse_args()
    rerouteInterval = args.reroute_interval
    setLogLevel('info')
    try:
        startNetwork()
    finally:
        cleanMininet() # also tear the network down if the experiment crashed
