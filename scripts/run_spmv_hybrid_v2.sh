#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --job-name=hybrid_v2_spmv
#SBATCH --output=hybrid_v2_spmv-%j.out
#SBATCH --error=hybrid_v2_spmv-%j.err
#SBATCH --nodelist=edu01

# Define executable and base directory
EXEC=~/GPU-Computing-2025-256137/bin/spmv_gpu_hybrid_v2_csr.exec
DATA_DIR=~/GPU-Computing-2025-256137/data

# Kernel parameters (see src/spmv_gpu_hybrid_v2_csr.cu); override on the command line:
#   sbatch scripts/run_spmv_hybrid_v2.sh 512 4096
BLOCK_SIZE=${1:-256}
HUGE_THRESHOLD=${2:-2048}

# Print header for results
echo "=================================================="
echo "Hybrid V2 SpMV Benchmark Results"
echo "=================================================="
echo "Started at: $(date)"
echo ""

# Define datasets to test
DATASETS=(
  "662_bus/662_bus.mtx"
  "Goodwin_127/Goodwin_127.mtx"
  "ML_Geer/ML_Geer.mtx"
  "Zd_Jac3_db/Zd_Jac3_db.mtx"
  "mawi_201512020330/mawi_201512020330.mtx"
  "CurlCurl_4/CurlCurl_4.mtx"
)

nvidia-smi

# Run benchmark for each dataset
for dataset in "${DATASETS[@]}"; do
  echo "------------------------------------------------"
  echo "Testing dataset: $dataset"
  echo "------------------------------------------------"
  echo "Config: block_size=$BLOCK_SIZE huge_threshold=$HUGE_THRESHOLD flush_l2=0"
  srun $EXEC $DATA_DIR/$dataset $BLOCK_SIZE $HUGE_THRESHOLD 0
  echo ""
  echo "Config: block_size=$BLOCK_SIZE huge_threshold=$HUGE_THRESHOLD flush_l2=1 (cold L2)"
  srun $EXEC $DATA_DIR/$dataset $BLOCK_SIZE $HUGE_THRESHOLD 1
  echo ""
done

echo "=================================================="
echo "Benchmark completed at: $(date)"
echo "=================================================="