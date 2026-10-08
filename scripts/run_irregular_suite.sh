#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01
#SBATCH --job-name=irreg_spmv
#SBATCH --output=irregular_spmv-%j.out
#SBATCH --error=irregular_spmv-%j.err
#
# Usage: sbatch scripts/run_irregular_suite.sh <impl> <batch>
#   impl:  adaptive | cusparse
#   batch: 1 | 2 | 3

IMPL=${1:-adaptive}
BATCH=${2:-1}
ROOT=~/cuda-SpMV
DATA_DIR=$ROOT/data
# shellcheck source=/dev/null
source "$ROOT/scripts/irregular_datasets.sh"

case "$IMPL" in
  adaptive) EXEC=$ROOT/bin/spmv_gpu_hybrid_adaptive_csr.exec; LABEL="Hybrid Adaptive" ;;
  cusparse) EXEC=$ROOT/test/cusparse.exec;                    LABEL="cuSPARSE" ;;
  *) echo "Unknown impl $IMPL (use adaptive|cusparse)"; exit 1 ;;
esac

case "$BATCH" in
  1) DATASETS=("${IRREGULAR_BATCH1[@]}") ;;
  2) DATASETS=("${IRREGULAR_BATCH2[@]}") ;;
  3) DATASETS=("${IRREGULAR_BATCH3[@]}") ;;
  *) echo "Unknown batch $BATCH"; exit 1 ;;
esac

echo "=================================================="
echo "Irregular SpMV suite — $LABEL (batch $BATCH)"
echo "Started at: $(date)"
echo "=================================================="
nvidia-smi
echo ""

for dataset in "${DATASETS[@]}"; do
  echo "------------------------------------------------"
  echo "Testing dataset: $dataset"
  echo "------------------------------------------------"
  if [[ ! -f "$DATA_DIR/$dataset" ]]; then
    echo "MISSING $DATA_DIR/$dataset — skip"
    continue
  fi
  srun $EXEC $DATA_DIR/$dataset
  echo ""
done

echo "=================================================="
echo "Completed at: $(date)"
echo "=================================================="
