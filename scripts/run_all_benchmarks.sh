#!/bin/bash
# filepath: /home/elia.gatti/GPU-Computing-2025-256137/run_all_benchmarks.sh

echo "=================================================="
echo "Starting full SpMV benchmarking suite"
echo "Started at: $(date)"
echo "=================================================="

# Path to the workspace
WORKSPACE=~/GPU-Computing-2025-256137

# Define all benchmark scripts to run
BENCHMARK_SCRIPTS=(
  "cpu_simple_run.sh"
  "cpu_ilp_run.sh"
  "run_spmv_hybrid_adaptive.sh"
  "run_spmv_hybrid_v2.sh"
  "run_cusparse.sh"
)

# Run each benchmark script and capture its job ID
for script in "${BENCHMARK_SCRIPTS[@]}"; do
  echo "Submitting benchmark: $script"
  
  # Submit the job and capture its ID
  JOB_ID=$(sbatch "$WORKSPACE/scripts/$script" | awk '{print $4}')
  
  if [ -n "$JOB_ID" ]; then
    echo "  Submitted as job $JOB_ID"
    
    # Store job IDs to track them later
    JOB_IDS+=("$JOB_ID")
  else
    echo "  Failed to submit job"
  fi
  
  # Add a small delay between submissions to avoid race conditions
  sleep 1
done

echo ""
echo "All benchmark jobs submitted. Job IDs: ${JOB_IDS[*]}"
echo ""
echo "To monitor progress, you can use:"
echo "  squeue -u \$USER"
echo ""
echo "Results will be available in the following files:"
echo "  spmv_benchmark-*.out (output files)"
echo "  spmv_benchmark-*.err (error files)"
echo ""
echo "=================================================="
echo "Benchmarking suite scheduled at: $(date)"
echo "=================================================="