#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --job-name=hybrid_adaptive_spmv
#SBATCH --output=hybrid_adaptive_spmv-%j.out
#SBATCH --error=hybrid_adaptive_spmv-%j.err
#SBATCH --nodelist=edu01

EXEC=~/cuda-SpMV/bin/spmv_gpu_hybrid_adaptive_csr.exec
DATA_DIR=~/cuda-SpMV/data
# shellcheck source=/dev/null
source ~/cuda-SpMV/scripts/irregular_datasets.sh
DATASETS=("${IRREGULAR_CORE[@]}")

echo "=================================================="
echo "SpMV Benchmark Results (Hybrid Adaptive)"
echo "=================================================="
echo "Started at: $(date)"
echo ""
nvidia-smi

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
echo "Benchmark completed at: $(date)"
echo "=================================================="
