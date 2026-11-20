#!/usr/bin/env bash
# =====================================================================
#  EditFinder - one-shot pipeline
#  Stage 1: pairwise alignment scoring
#  Stage 2: CNN-based edit / SNP classification
# =====================================================================
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_editfinder.sh <INPUT_FA> <OUT_DIR> <MODEL_PATH> <DEVELOPMENT_FA>

Arguments:
  INPUT_FA         Query FASTA (one record = one candidate editing site)
  OUT_DIR          Output directory (scores + BED files written here)
  MODEL_PATH       Path to the .pth checkpoint
  DEVELOPMENT_FA   Development  FASTA used by Stage 1 scoring

Example:
  bash run_editfinder.sh \
      /test_input/test_input.fa \
      /model/final_model_seed12345.pth \
      /model/development.fa
EOF
    exit 1
}

[[ $# -ne 4 ]] && usage

INPUT_FA=$1
OUT_DIR=$2
MODEL_PATH=$3
DEVELOPMENT_FA=$4

# ---------- sanity checks ----------
[[ -f "$INPUT_FA"       ]] || { echo "ERROR: input FASTA not found: $INPUT_FA" >&2; exit 1; }
[[ -f "$MODEL_PATH"     ]] || { echo "ERROR: model not found: $MODEL_PATH" >&2; exit 1; }
[[ -f "$DEVELOPMENT_FA" ]] || { echo "ERROR: DEVELOPMENT_FA not found: $DEVELOPMENT_FA" >&2; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT_DIR"
SCORE_FILE="$OUT_DIR/alignment_scores.csv"
PRED_DIR="$OUT_DIR/predictions"
mkdir -p "$PRED_DIR"

# ---------- Stage 1: alignment scoring ----------
echo "[EditFinder] Stage 1/2 - pairwise alignment scoring ..."
python "$SCRIPT_DIR/NW_pair_seq_alignment.py" \
    "$INPUT_FA" \
    "$OUT_DIR" \
    "$DEVELOPMENT_FA"

[[ -f "$SCORE_FILE" ]] || { echo "ERROR: Stage 1 did not produce $SCORE_FILE" >&2; exit 1; }

# ---------- Stage 2: CNN prediction ----------
echo "[EditFinder] Stage 2/2 - CNN prediction ..."
python "$SCRIPT_DIR/EditFinder_predict.py" \
    --fasta       "$INPUT_FA" \
    --scores      "$SCORE_FILE" \
    --model       "$MODEL_PATH" \
    --output_dir  "$PRED_DIR" \
    --threshold   0.5 \
    --batch_size  1024

echo
echo "[EditFinder] Done."
echo "  edit BED : $PRED_DIR/edit_predictions.bed"
echo "  snp  BED : $PRED_DIR/snp_predictions.bed"