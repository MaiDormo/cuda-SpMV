#!/bin/bash
# filepath: /home/elia.gatti/cuda-SpMV/test/compile_perf.sh

if [ $# -eq 0 ]; then
  echo "Usage: $0 filename.cu"
  exit 1
fi

module load CUDA/12.5.0

filename=$(basename -- "$1")
name="${filename%.*}"

# Maximum performance compilation
nvcc -O3 --use_fast_math --gpu-architecture=sm_80 -m64 \
     --maxrregcount=255 \
     --ptxas-options=--warn-on-spills,--optimize-float-atomics \
     -Xcompiler "-Ofast -march=native -mtune=native -funroll-loops -ffast-math -fopenmp -DNDEBUG" \
     -Xptxas -O3,--def-load-cache=ca,--def-store-cache=wb \
     --relocatable-device-code=false \
     --restrict \
     --fmad=true \
     -I../include \
     -o "$name.exec" "$1" \
     ../lib/read_file_lib.c ../lib/coo_to_csr.c ../lib/spmv_utils.c \
     -lcusparse -lcudart

echo "Compiled $1 into $name.exec with maximum performance optimizations"