#!/bin/bash
# One-time environment setup (Fedora). Run as your normal user: ./setup.sh
# Steps that need root are run through sudo.
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPT="$HOME/opt"
JDK8="$OPT/jdk8"
MAVEN="$OPT/apache-maven-3.9.9"
FLOODLIGHT="$HOME/floodlight"
mkdir -p "$OPT"

# --- System packages (tc for TCLink, OVS, iperf3, ffmpeg, Python 3.11) ---
sudo dnf install -y iproute-tc openvswitch iperf3 ffmpeg-free python3.11 gcc make git
sudo systemctl enable --now openvswitch
sudo modprobe -a openvswitch sch_htb sch_netem

# --- Python virtualenv ---
[ -d "$PROJECT_DIR/.venv" ] || python3.11 -m venv "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# --- Mininet's mnexec helper (the pip package does not build it) ---
[ -d "$OPT/mininet" ] || git clone --depth 1 -b 2.3.1b4 https://github.com/mininet/mininet.git "$OPT/mininet"
make -C "$OPT/mininet" mnexec
sudo install -m 755 "$OPT/mininet/mnexec" /usr/bin/mnexec

# --- JDK 8 + Maven (Floodlight does not build or run on newer JDKs) ---
if [ ! -d "$JDK8" ]; then
    curl -sSL -o "$OPT/jdk8.tar.gz" "https://api.adoptium.net/v3/binary/latest/8/ga/linux/x64/jdk/hotspot/normal/eclipse"
    mkdir -p "$JDK8" && tar xzf "$OPT/jdk8.tar.gz" -C "$JDK8" --strip-components=1 && rm "$OPT/jdk8.tar.gz"
fi
[ -d "$MAVEN" ] || curl -sSL https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz | tar xz -C "$OPT"

# --- Floodlight (master: has /wm/routing/paths, /wm/routing/metric, /wm/staticentrypusher) ---
[ -d "$FLOODLIGHT" ] || git clone https://github.com/floodlight/floodlight.git "$FLOODLIGHT"
cd "$FLOODLIGHT"
git submodule update --init
export JAVA_HOME="$JDK8" PATH="$JDK8/bin:$MAVEN/bin:$PATH"
# openflowj 3.3.0-SNAPSHOT is no longer published; install the jar bundled in lib/
mvn -q -B install:install-file -Dfile=lib/openflowj-3.3.0-SNAPSHOT.jar \
    -DgroupId=org.projectfloodlight -DartifactId=openflowj -Dversion=3.3.0-SNAPSHOT -Dpackaging=jar
mvn -q -B package -Dmaven.test.skip=true

echo "Setup complete. Start Floodlight with: $PROJECT_DIR/start_floodlight.sh"
