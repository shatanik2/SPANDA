#!/bin/bash
# ============================================================
# SPANDA: full pipeline (Stages 1, 2 and 3) for one star.
# Usage:  ./run_single_star.sh <KIC> <NUMAX_muHz> <DNU_muHz> <PSD_DIR> [PSD_PATTERN] [WORK_DIR]
#   PSD_PATTERN : PSD file name inside PSD_DIR, with {kic} as placeholder (default: {kic}.npy)
#   WORK_DIR    : output folder (default: ./runs/<KIC>)
# ============================================================
set -euo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <KIC> <NUMAX_muHz> <DNU_muHz> <PSD_DIR> [PSD_PATTERN] [WORK_DIR]"
    exit 1
fi

KIC="$1"
NUMAX="$2"
DNU="$3"
PSD_DIR="$(cd "$4" && pwd)"
DEFAULT_PATTERN='{kic}.npy'
PSD_PATTERN="${5:-$DEFAULT_PATTERN}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${6:-./runs/${KIC}}"

PSD_FILE="${PSD_DIR}/${PSD_PATTERN//\{kic\}/$KIC}"
if [ ! -f "$PSD_FILE" ]; then
    echo "ERROR: PSD file not found: $PSD_FILE"
    exit 1
fi

mkdir -p "$WORK"
cd "$WORK"
mkdir -p logs_stage1 logs_stage2 logs_l0 logs_l2

printf "KIC,numax,Delnu\n%s,%s,%s\n" "$KIC" "$NUMAX" "$DNU" > init.csv

step() { echo; echo "=== $1 ==="; }
completed() { grep -q "$2" "$1" 2>/dev/null; }

step "Stage 1: background fit"
python3 "$ROOT/Stage1/stage1_background_fit.py" "$KIC" --catalog init.csv \
    --psd-dir "$PSD_DIR" --psd-pattern "$PSD_PATTERN" > "logs_stage1/kic_${KIC}.log" 2>&1
completed "logs_stage1/kic_${KIC}.log" "Total runtime:" || { echo "Stage 1 failed; see logs_stage1/kic_${KIC}.log"; exit 1; }
python3 "$ROOT/Stage1/make_stage1_summary.py" --logdir logs_stage1 --catalog init.csv \
    --psd-dir "$PSD_DIR" --psd-pattern "$PSD_PATTERN" > logs_stage1/summary.out 2>&1
S1="stage1_summary_1/stage1_summary_1.csv"
[ -f "$S1" ] || { echo "Stage 1 summary failed; see logs_stage1/summary.out"; exit 1; }
grep -E "Mean of residuals \(Outside\)" "logs_stage1/kic_${KIC}.log"
python3 - "$S1" <<'EOF'
import sys, pandas as pd
r = pd.read_csv(sys.argv[1]).iloc[0]
ok = abs(1 - r.mean_res_outside) <= 0.02 and abs(1 - r.std_res_outside) <= 0.05
print("Residual selection (|1-mean| <= 0.02, |1-std| <= 0.05 outside the power excess):",
      "passed" if ok else "NOT passed; Stages 2 and 3 are run, but treat their results with care")
EOF

step "Stage 2a: epsilon_p"
python3 "$ROOT/Stage2/stage2a_epsilon_p.py" "$KIC" --stage1-csv "$S1" \
    --psd-dir "$PSD_DIR" --psd-pattern "$PSD_PATTERN" --plot > logs_stage2/stage2a_${KIC}.log 2>&1 \
    || { echo "Stage 2a failed; see logs_stage2/stage2a_${KIC}.log"; exit 1; }
tail -1 logs_stage2/stage2a_${KIC}.log
python3 "$ROOT/Stage2/stage2_collect_epsilon_p.py" > logs_stage2/collect.out 2>&1

step "Stage 2b: radial-mode frequencies and heights"
python3 "$ROOT/Stage2/stage2b_radial_modes.py" "$KIC" --stage1-csv "$S1" \
    --psd-dir "$PSD_DIR" --psd-pattern "$PSD_PATTERN" --plot > logs_stage2/stage2b_${KIC}.log 2>&1 \
    || { echo "Stage 2b failed; see logs_stage2/stage2b_${KIC}.log"; exit 1; }
[ -f "p_mode_frequencies/${KIC}_p_frequencies.npy" ] || { echo "Stage 2b wrote no modes"; exit 1; }
grep -E "Saved [0-9]+ mode measurements" logs_stage2/stage2b_${KIC}.log

step "Stage 3a: radial modes"
python3 "$ROOT/Stage3/stage3a_l0_fit.py" "$KIC" "$S1" p_mode_frequencies \
    --psd-dir "$PSD_DIR" --psd-pattern "$PSD_PATTERN" > "logs_l0/kic_${KIC}.log" 2>&1
completed "logs_l0/kic_${KIC}.log" "Total Pipeline Time:" || { echo "Stage 3a failed; see logs_l0/kic_${KIC}.log"; exit 1; }
python3 "$ROOT/Stage3/make_stage3a_summary.py" --logdir logs_l0 > logs_l0/summary.out 2>&1
S3A="stage3a_summary_1/stage3a_summary_1.csv"
[ -f "$S3A" ] || { echo "Stage 3a summary failed; see logs_l0/summary.out"; exit 1; }
grep -E "Identified|Max split R-hat|Gamma0 \(MAP\)" "logs_l0/kic_${KIC}.log"

step "Stage 3b: quadrupole modes"
python3 "$ROOT/Stage3/stage3b_l2_fit.py" "$KIC" "$S1" "$S3A" \
    --psd-dir "$PSD_DIR" --psd-pattern "$PSD_PATTERN" > "logs_l2/kic_${KIC}.log" 2>&1
completed "logs_l2/kic_${KIC}.log" "Total Pipeline Time:" || { echo "Stage 3b failed; see logs_l2/kic_${KIC}.log"; exit 1; }
python3 "$ROOT/Stage3/make_stage3b_summary.py" --logdir logs_l2 > logs_l2/summary.out 2>&1
[ -f "stage3b_summary_1/stage3b_summary_1.csv" ] || { echo "Stage 3b summary failed; see logs_l2/summary.out"; exit 1; }
grep -E "Max split R-hat|Fixed Gamma0 status" "logs_l2/kic_${KIC}.log"

step "Done"
echo "Results for KIC ${KIC} are in $(pwd):"
echo "  stage1_summary_1/stage1_summary_1.csv     background fit"
echo "  stage3a_summary_1/stage3a_summary_1.csv   l=0 fit"
echo "  stage3b_summary_1/stage3b_summary_1.csv   l=2 fit"
echo "  stage1_fit_pdfs/, stage2a_plots/, stage2b_plots/, stage3a_fit_pdfs/, stage3b_fit_pdfs/   plots"
echo "  logs_*/                                    full logs of every step"
