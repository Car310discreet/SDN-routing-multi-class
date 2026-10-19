#!/bin/bash
# Reset Mininet state (OVS switches, veth links, leftover processes) after a crashed run
MN="$(dirname "$0")/../.venv/bin/mn"
[ -x "$MN" ] || MN=mn
"$MN" -c
