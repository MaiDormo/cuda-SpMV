#!/bin/bash
# filepath: /home/elia.gatti/cuda-SpMV/extract_hybrid_sweep_data.sh
# Script to extract hybrid adaptive SpMV configuration sweep data
# Usage: ./extract_hybrid_sweep_data.sh [input_file] [output_file.csv]

INPUT_FILE="${1:-hybrid_adaptive_sweep-*.out}"
OUTPUT_CSV="${2:-hybrid_sweep_results.csv}"

# Create CSV header
echo "Dataset,Matrix_Name,Rows,Columns,NNZ,Mean_NNZ_Per_Row,Std_Dev_NNZ,Max_NNZ_Per_Row,Empty_Rows,Empty_Rows_Percent,Config_Threads,Config_Threshold,Short_Rows,Short_Rows_Percent,Long_Rows,Long_Rows_Percent,Launch_Blocks,Launch_Short_Blocks,Launch_Long_Blocks,Execution_Time_Seconds,Bandwidth_GB_s,Performance_GFLOPS,Arithmetic_Intensity" > "$OUTPUT_CSV"

# Function to process the sweep output file
process_sweep_file() {
    local file=$1
    echo "Processing $file..."
    
    awk '
    BEGIN {
        dataset = ""; matrix_name = "";
        rows = ""; cols = ""; nnz = ""; mean_nnz = ""; std_dev = ""; max_nnz = "";
        empty_rows = ""; empty_rows_percent = "";
        config_threads = ""; config_threshold = "";
        short_rows = ""; short_rows_percent = ""; long_rows = ""; long_rows_percent = "";
        launch_blocks = ""; launch_short_blocks = ""; launch_long_blocks = "";
        time = ""; bandwidth = ""; gflops = ""; arithmetic_intensity = "";
        in_config = 0;
    }
    
    # Dataset detection
    /^Testing dataset:/ {
        if (in_config && time != "") {
            output_csv_line();
        }
        
        dataset = $3;
        gsub(/\/.*\.mtx/, "", dataset);
        matrix_name = dataset;
        
        reset_variables();
    }
    
    # Configuration detection
    /^CONFIGURATION: THREADS=/ {
        if (in_config && time != "") {
            output_csv_line();
        }
        
        match($0, /THREADS=([0-9]+) THRESHOLD=([0-9]+)/, arr);
        config_threads = arr[1];
        config_threshold = arr[2];
        in_config = 1;
        
        # Reset performance variables for new config
        time = ""; bandwidth = ""; gflops = ""; arithmetic_intensity = "";
    }
    
    # Matrix analysis
    /^  Mean NNZ per row:/ { mean_nnz = $5; }
    /^  Std deviation:/ { std_dev = $3; }
    /^  Max NNZ per row:/ { max_nnz = $5; }
    /^  Empty rows:/ { 
        empty_rows = $3;
        match($0, /\(([0-9.]+)%\)/, arr);
        empty_rows_percent = arr[1];
    }
    
    # Matrix dimensions
    /^Dimensions:/ {
        match($0, /([0-9]+) x ([0-9]+), NNZ: ([0-9]+)/, arr);
        rows = arr[1];
        cols = arr[2];
        nnz = arr[3];
    }
    
    # Row classification
    /^  Short rows:/ {
        short_rows = $3;
        match($0, /\(([0-9.]+)%\)/, arr);
        short_rows_percent = arr[1];
    }
    /^  Long rows:/ {
        long_rows = $3;
        match($0, /\(([0-9.]+)%\)/, arr);
        long_rows_percent = arr[1];
    }
    
    # Launch configuration
    /^Launch config:/ {
        match($0, /([0-9]+) blocks \(([0-9]+) short \+ ([0-9]+) long\)/, arr);
        launch_blocks = arr[1];
        launch_short_blocks = arr[2];
        launch_long_blocks = arr[3];
    }
    
    # Performance metrics
    /^Average execution time:/ { time = $4; }
    /^Memory bandwidth \(estimated\):/ { bandwidth = $4; }
    /^Computational performance:/ { gflops = $3; }
    /^Arithmetic intensity:/ { arithmetic_intensity = $3; }
    
    # End of file processing
    END {
        if (in_config && time != "") {
            output_csv_line();
        }
    }
    
    function reset_variables() {
        time = ""; bandwidth = ""; gflops = ""; arithmetic_intensity = "";
        short_rows = ""; short_rows_percent = ""; long_rows = ""; long_rows_percent = "";
        launch_blocks = ""; launch_short_blocks = ""; launch_long_blocks = "";
    }
    
    function output_csv_line() {
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
               dataset, matrix_name, rows, cols, nnz, mean_nnz, std_dev, max_nnz,
               empty_rows, empty_rows_percent, config_threads, config_threshold,
               short_rows, short_rows_percent, long_rows, long_rows_percent,
               launch_blocks, launch_short_blocks, launch_long_blocks,
               time, bandwidth, gflops, arithmetic_intensity;
    }
    ' "$file" >> "$OUTPUT_CSV"
}

# Find and process the sweep output file
if [ -f "$INPUT_FILE" ]; then
    process_sweep_file "$INPUT_FILE"
else
    # Look for pattern match
    sweep_files=$(find . -name "hybrid_adaptive_sweep-*.out" | head -1)
    if [ -n "$sweep_files" ]; then
        for file in $sweep_files; do
            if [ -f "$file" ]; then
                process_sweep_file "$file"
            fi
        done
    else
        echo "No hybrid adaptive sweep output files found!"
        echo "Looking for files matching pattern: hybrid_adaptive_sweep-*.out"
        exit 1
    fi
fi

echo ""
echo "Data extraction complete!"
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Summary:"
echo "Total records: $(tail -n +2 "$OUTPUT_CSV" | wc -l)"
echo "Unique datasets: $(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f1 | sort -u | wc -l)"
echo "Unique thread configurations: $(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f11 | sort -u | wc -l)"
echo "Unique threshold configurations: $(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f12 | sort -u | wc -l)"
echo ""
echo "Preview of first 5 records:"
head -6 "$OUTPUT_CSV" | column -t -s','