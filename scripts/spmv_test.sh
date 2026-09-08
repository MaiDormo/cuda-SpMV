#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=hybrid_adaptive_sweep
#SBATCH --output=hybrid_adaptive_sweep-%j.out
#SBATCH --error=hybrid_adaptive_sweep-%j.err
#SBATCH --nodelist=edu01

# Define executable and base directory
EXEC=~/cuda-SpMV/bin/spmv_mawi_test.exec
DATA_DIR=~/cuda-SpMV/data

# Print header for results
echo "=================================================="
echo "Hybrid Adaptive SpMV Configuration Sweep"
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

# Define configuration parameters to test
THREAD_CONFIGS=(128 256 512 1024)
THRESHOLD_CONFIGS=(16 32 64 128 256)

nvidia-smi

# Run benchmark for each configuration on each dataset
for dataset in "${DATASETS[@]}"; do
  echo "=================================================="
  echo "Testing dataset: $dataset"
  echo "=================================================="
  
  for threads in "${THREAD_CONFIGS[@]}"; do
    for threshold in "${THRESHOLD_CONFIGS[@]}"; do
      echo "------------------------------------------------------------"
      echo "CONFIGURATION: THREADS=$threads THRESHOLD=$threshold"
      echo "Dataset: $dataset"
      echo "------------------------------------------------------------"
      srun $EXEC $DATA_DIR/$dataset $threads $threshold
      echo ""
    done
  done
  
  echo "Completed all configurations for: $dataset"
  echo ""
done

echo "=================================================="
echo "All configurations completed at: $(date)"
echo "=================================================="