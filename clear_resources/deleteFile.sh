#!/bin/bash
# Script to delete a specified file safely
if [ -n "$1" ] && [ -f "$1" ]; then
    rm -f "$1"
fi
