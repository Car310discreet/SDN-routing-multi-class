#!/bin/bash
# Start the Floodlight controller (OpenFlow on 6653, REST API on 8080)
cd "$HOME/floodlight" && exec "$HOME/opt/jdk8/bin/java" -jar target/floodlight.jar
