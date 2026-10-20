#!/usr/bin/env bash
# Rebuild report/GEANT2_Technical_Report.pdf from geant2/REPORT.md (needs pandoc and LuaLaTeX).
# Regenerates report/figures/ first (report/make_figures.py).
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -W ignore report/make_figures.py
pandoc geant2/REPORT.md -o report/GEANT2_Technical_Report.pdf --pdf-engine=lualatex \
  --shift-heading-level-by=-1 --columns=60 --resource-path=geant2 --lua-filter=report/breakable_code.lua -H report/pdf_header.tex \
  -V documentclass=article -V papersize=a4 -V geometry:margin=2.2cm -V fontsize=10pt \
  -V mainfont="Noto Serif" -V mainfontfallback="Symbola:" -V sansfont="Noto Sans" \
  -V monofont="DejaVu Sans Mono" -V monofontfallback="Symbola:" \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue
echo "Wrote report/GEANT2_Technical_Report.pdf"
