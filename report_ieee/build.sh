#!/usr/bin/env bash
# build.sh - regenerate the report figures and compile main.tex to main.pdf.
# Needs a LaTeX engine with IEEEtran: tectonic (pass its path as TECTONIC=...) or pdflatex
# (sudo apt install texlive-latex-extra texlive-publishers). Overleaf also compiles this folder as is.
set -euo pipefail
cd "$(dirname "$0")"
../.venv/bin/python make_report_figures.py
if [ -n "${TECTONIC:-}" ]; then "$TECTONIC" main.tex
elif command -v tectonic >/dev/null; then tectonic main.tex
else pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
fi
echo "report_ieee/main.pdf"
