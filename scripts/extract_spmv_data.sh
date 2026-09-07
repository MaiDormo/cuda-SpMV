#!/bin/bash

OUTPUT_CSV="${1:-spmv_results_minimal.csv}"

echo "Platform,Implementation,Dataset,Matrix_Name,Rows,Columns,NNZ,Mean_NNZ_Per_Row,Execution_Time_Seconds,Bandwidth_GB_s,Performance_GFLOPS" > "$OUTPUT_CSV"

process_benchmark_file() {
    local file=$1
    local filename=$(basename "$file")

    # Platform detection
    local platform="CPU"
    if grep -q -E "NVIDIA-SMI|nvidia-smi|CUDA|sm_|Device Query|Multiprocessors|Global Memory.*MBytes" "$file"; then
        platform="GPU"
    fi

    # Implementation detection (order matters)
    local implementation="unknown"
    if [[ "$filename" =~ hybrid_v2 ]] || grep -q "Hybrid V2 CSR" "$file"; then
        implementation="gpu_hybrid_v2"
    elif [[ "$filename" =~ gpu_hybrid_adaptive ]] || grep -q "Hybrid Adaptive CSR" "$file"; then
        implementation="gpu_hybrid_adaptive"
    elif [[ "$filename" =~ gpu_adaptive ]] || grep -q "Adaptive CSR" "$file"; then
        implementation="gpu_adaptive"
    elif [[ "$filename" =~ cpu_simple ]] || grep -q "CPU Simple CSR" "$file"; then
        implementation="cpu_simple"
    elif [[ "$filename" =~ cpu_ilp ]] || grep -q "CPU ILP CSR" "$file"; then
        implementation="cpu_ilp"
    elif [[ "$filename" =~ gpu_simple ]] || grep -q "Simple CSR" "$file"; then
        implementation="gpu_simple"
    elif [[ "$filename" =~ gpu_vector ]] || grep -q "Vector CSR" "$file"; then
        implementation="gpu_vector"
    elif [[ "$filename" =~ gpu_value_sequential ]] || grep -q "Value Sequential|CSR.*Sequential" "$file"; then
        implementation="gpu_value_sequential"
    elif [[ "$filename" =~ gpu_value_blocked ]] || grep -q "Value.*Blocked|Blocked.*CSR" "$file"; then
        implementation="gpu_value_blocked"
    elif [[ "$filename" =~ cusparse ]] || grep -q "cuSPARSE" "$file"; then
        implementation="cusparse"
    fi

    awk -v platform="$platform" -v impl="$implementation" '
    BEGIN {
        dataset = ""; matrix_name = ""; rows = ""; cols = ""; nnz = ""; mean_nnz = "";
        time = ""; bandwidth = ""; gflops = "";
        in_dataset = 0;
    }
    # Dataset detection
    /^Testing dataset:/ {
        if (in_dataset && rows && time) output_csv_line();
        dataset_path = $3;
        split(dataset_path, path_parts, "/");
        dataset = path_parts[length(path_parts)];
        gsub(/\.mtx.*/, "", dataset);
        matrix_name = dataset;
        reset_vars();
        in_dataset = 1;
        next;
    }
    # Matrix file detection
    /^Matrix file:/ {
        matrix_file = substr($0, index($0, ":") + 2);
        gsub(/^[ \t]+/, "", matrix_file);
        split(matrix_file, path_parts, "/");
        dataset = path_parts[length(path_parts)];
        gsub(/\.mtx.*/, "", dataset);
        matrix_name = dataset;
        in_dataset = 1;
        next;
    }
    # Matrix size: "Matrix size: 662 x 662 with 2,474 non-zero elements"
    /^Matrix size:/ {
        if ($0 ~ /Matrix size:.*x.*with.*non-zero/) {
            match($0, /([0-9,]+) x ([0-9,]+) with ([0-9,]+) non-zero/, arr);
            rows = arr[1]; gsub(/,/, "", rows);
            cols = arr[2]; gsub(/,/, "", cols);
            nnz = arr[3]; gsub(/,/, "", nnz);
        }
    }
    # Matrix stats
    /^  Mean NNZ per row:/ { mean_nnz = $5; }
    /^Average NNZ per Row:/ { if (mean_nnz == "") mean_nnz = $5; }
    # Performance metrics
    /^Average execution time:/ {
        time = $4;
        if ($5 == "ms" || $5 == "milliseconds") time = time / 1000.0;
        else if ($5 == "μs" || $5 == "microseconds") time = time / 1000000.0;
        gsub(/[^0-9.eE+-]/, "", time);
    }
    # --- CHANGED SECTION: Robust bandwidth extraction ---
    ($0 ~ /Memory bandwidth/ || $0 ~ /^Bandwidth:/) {
        match($0, /: *([0-9.eE+-]+)/, arr);
        if (arr[1] != "") bandwidth = arr[1];
    }
    (/^Computational performance:/ || /^Performance:/) {
        gflops = $3;
        gsub(/[^0-9.eE+-]/, "", gflops);
    }
    END { if (in_dataset && rows && time) output_csv_line(); }
    function reset_vars() {
        rows = ""; cols = ""; nnz = ""; mean_nnz = "";
        time = ""; bandwidth = ""; gflops = "";
    }
    function output_csv_line() {
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
            platform, impl, dataset, matrix_name, rows, cols, nnz, mean_nnz, time, bandwidth, gflops;
    }
    ' "$file" >> "$OUTPUT_CSV"
}

echo "Searching for SpMV benchmark output files..."

benchmark_files=$(find . -maxdepth 1 -name "*.out" -type f | grep -E "(spmv|cpu_simple|cpu_ilp|gpu_simple|gpu_vector|gpu_adaptive|gpu_hybrid_adaptive|hybrid_v2|gpu_value_sequential|gpu_value_blocked|cublas|adaptive_spmv|hybrid_adaptive_spmv)" | sort)

if [ -z "$benchmark_files" ]; then
    echo "No benchmark output files found in current directory!"
    echo "Available .out files:"
    ls -la *.out 2>/dev/null || echo "No .out files found"
    exit 1
fi

echo "Found $(echo "$benchmark_files" | wc -l) benchmark files:"
echo "$benchmark_files"
echo ""

for file in $benchmark_files; do
    if [ -f "$file" ]; then
        process_benchmark_file "$file"
    fi
done

awk '!seen[$0]++' "$OUTPUT_CSV" > "${OUTPUT_CSV}.tmp" && mv "${OUTPUT_CSV}.tmp" "$OUTPUT_CSV"

echo ""
echo "Data extraction complete!"5
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
total_records=$(tail -n +2 "$OUTPUT_CSV" | wc -l)
unique_implementations=$(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f2 | sort -u | wc -l)
unique_datasets=$(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f3 | sort -u | wc -l)

echo "Total records: $total_records"
echo "Unique implementations: $unique_implementations"
echo "Unique datasets: $unique_datasets"
echo ""

if [ $total_records -gt 0 ]; then
    echo "Preview of first 5 records:"
    head -n 6 "$OUTPUT_CSV" | column -t -s',' | head -n 6
else
    echo "Warning: No data records were extracted!"
    echo "Check that your benchmark output files contain the expected format."
fi