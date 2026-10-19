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

net = None
activeThreadList = []
trafficTime = 0
# Floodlight routing metric for the baseline run:
#   hopcount    -> default Floodlight routing (paper baseline 1)
#   latency     -> QoS-aware routing (paper baseline 2)
#   utilization -> QoS-aware routing using link bandwidth utilization
routingMetric = "latency"

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
    trafficTime = 900 # must outlast the whole experiment so background traffic is present during measurement and video
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
    sleep(7) # Allow Floodlight LLDP discovery to discover all switch links

    # Configure controller
    info('[INFO]****** Configuring Controller *****\n')
    floodlightRestApi.deleteAllFlows() # static flows persist in Floodlight across runs
    info(f'[INFO]****** Routing metric: {routingMetric} *****\n')
    if routingMetric == "utilization":
        floodlightRestApi.enableSwitchStats()
    floodlightRestApi.setRoutingMetric(routingMetric)

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
    psnr_val, ssim_first = None, None
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

    # Persist the video quality next to the ping/throughput outputs (used by the comparison plots)
    with open(f"output_{routingMetric}_video.json", "w") as videoFile:
        json.dump({"psnr_db": psnr_val, "ssim": ssim_first}, videoFile)

    time_2 = perf_counter()
    print("Time passed: ", time_2 - start)


    info(f'[INFO]****** Active thread count: {threading.active_count()}\n')

    # info(f'[INFO]****** Video File Removing *****\n\n')
    # try:
    #     deleteFile("records/input.ts")
    # except Exception as e:
    #     print(f'Exception occurred: {e}')

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

    if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        info('[INFO]****** No GUI display detected; skipping video preview window *****\n')
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


def startStream(sender: str, receiver: str):
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

    with open("receiver.log", "w") as receiverLog, open("sender.log", "w") as senderLog:
        receiverProcess = receiverNode.popen(receiverCommand, stdout=receiverLog, stderr=subprocess.STDOUT)
        sleep(0.5)
        senderProcess = senderNode.popen(senderCommand, stdout=senderLog, stderr=subprocess.STDOUT)

        senderProcess.wait()
        sleep(2)
        info(f'[INFO]****** Stopping video receiver *****\n')
        receiverProcess.terminate() # ffmpeg finalizes the output file on SIGTERM
        try:
            receiverProcess.wait(timeout=10)
        except subprocess.TimeoutExpired:
            receiverProcess.kill()

    if previewProcess is not None:
        info(f'[INFO]****** Closing video preview window *****\n')
        try:
            previewProcess.terminate()
            previewProcess.wait(timeout=3)
        except subprocess.TimeoutExpired:
            previewProcess.kill()
        except Exception as exc:
            info(f'[WARN]****** Error closing preview window: {exc} *****\n')

def getPingStats(sender: str, receiver: str, getStats = False) -> tuple[float, float]:
    senderNode, receiverNode = net.getNodeByName(sender), net.getNodeByName(receiver)
    receiverIpv4 = receiverNode.IP()
    print("Receiver ipv4 -> ", receiverIpv4)

    if getStats == True:
        packetCount = "60"
        command = f"ping {receiverIpv4} -c {packetCount} > output_{routingMetric}_ping.txt"
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
    clientCommand = f"iperf3 -c {server.IP()} -p {port} -b {bandWidth} -R -t {time} > output_{routingMetric}_throughput.txt"

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
            if (flow.src_ip in ("10.0.0.1", "10.0.0.9") and flow.dst_ip in ("10.0.0.1", "10.0.0.9")):
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
    if len(sys.argv) > 1:
        routingMetric = sys.argv[1]
    setLogLevel('info')
    try:
        startNetwork()
    finally:
        cleanMininet() # also tear the network down if the experiment crashed
