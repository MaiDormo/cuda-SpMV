#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --job-name=cpu_simple_spmv_benchmark
#SBATCH --output=cpu_simple_spmv_benchmark-%j.out
#SBATCH --error=cpu_simple_spmv_benchmark-%j.err

# Define executable and base directory
EXEC=~/cuda-SpMV/bin/spmv_cpu_csr
DATA_DIR=~/cuda-SpMV/data

# Print header for results
echo "=================================================="
echo "SpMV Benchmark Results CPU"
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

lscpu | grep "Model name"

# Run benchmark for each dataset
for dataset in "${DATASETS[@]}"; do
  echo "------------------------------------------------"
  echo "Testing dataset: $dataset"
  echo "------------------------------------------------"
  srun $EXEC $DATA_DIR/$dataset
  echo ""
done

echo "=================================================="
echo "Benchmark completed at: $(date)"
echo "=================================================="