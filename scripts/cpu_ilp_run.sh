#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --job-name=cpu_ilp_spmv_benchmark
#SBATCH --output=cpu_ilp_spmv_benchmark-%j.out
#SBATCH --error=cpu_ilp_spmv_benchmark-%j.err

EXEC=~/cuda-SpMV/bin/spmv_cpu_csr_ilp
DATA_DIR=~/cuda-SpMV/data
# shellcheck source=/dev/null
source ~/cuda-SpMV/scripts/irregular_datasets.sh
DATASETS=("${IRREGULAR_CORE[@]}")

echo "=================================================="
echo "SpMV Benchmark Results CPU (ILP CSR)"
echo "=================================================="
echo "Started at: $(date)"
echo ""
lscpu | grep "Model name"

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
