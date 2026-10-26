#!/usr/bin/env bash
# Rebuild advanced/report/Advanced_Technical_Report.pdf from advanced/REPORT.md (needs pandoc and LuaLaTeX).
# Regenerates advanced/report/figures/ first (advanced/report/make_figures.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python -W ignore advanced/report/make_figures.py
pandoc advanced/REPORT.md -o advanced/report/Advanced_Technical_Report.pdf --pdf-engine=lualatex \
  --shift-heading-level-by=-1 --columns=60 --resource-path=advanced \
  --lua-filter=advanced/report/breakable_code.lua -H advanced/report/pdf_header.tex \
  -V documentclass=article -V papersize=a4 -V geometry:margin=2.2cm -V fontsize=10pt \
  -V mainfont="Noto Serif" -V mainfontfallback="Symbola:" -V sansfont="Noto Sans" \
  -V monofont="DejaVu Sans Mono" -V monofontfallback="Symbola:" \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue
echo "Wrote advanced/report/Advanced_Technical_Report.pdf"
