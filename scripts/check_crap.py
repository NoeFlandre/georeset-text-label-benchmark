"""Run the package's tested CRAP score gate."""

from __future__ import annotations

import sys

from georeset_text_label_benchmark.quality.crap import main

if len(sys.argv) != 2:
    raise SystemExit("usage: check_crap.py coverage.json")

raise SystemExit(main(sys.argv[1]))
