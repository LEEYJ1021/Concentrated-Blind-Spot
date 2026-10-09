#!/usr/bin/env bash
# Usage: RAIL_ROOT=/path/to/AI_Rail_OM [QUICK=1] bash run_all.sh
# RAIL_ROOT must contain Data/ (Korail CSVs). QUICK=1 is a smoke test only: never cite its output.
set -e
cd "$(dirname "$0")/codes"
for f in 00_build_data_bundle_v3.py 01_phase0_audit.py 02_main_reanalysis.py \
         03_supp_S_scope_rewiring_spatial.py 04_patch_P_spec_curve.py \
         05_supp_C_size_power_calibration.py 06_supp_D_statistic_comparison.py \
         07_supp_E_detectability.py; do
  echo "=== $f ==="; python "$f"
done
