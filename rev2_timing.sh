#!/bin/bash
# Rev. 2 timing on the same PC as Fig. 8. Run it ALONE (no ns-3 or other jobs running).
cd $HOME/AI-DTOPA
python runtime_scaling.py --out results/runtime_fix.csv
python rev2_latency.py
tar czf rev2_timing_results.tar.gz results/runtime_fix.csv results/rev2_latency.csv results/rev2_latency_runs.csv
echo "send me rev2_timing_results.tar.gz"
