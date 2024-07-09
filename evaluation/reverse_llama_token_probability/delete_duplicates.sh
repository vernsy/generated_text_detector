#!/bin/bash

input_file="/home/verena/master_thesis/data/e_llama_text_generation/error_analysis_springernature.jsonl"
output_file="/home/verena/master_thesis/data/e_llama_text_generation/error_analysis.tsv"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
        echo "Input file does not exist."
        exit 1
fi

# Counter to keep track of rows
counter=0

# Read the input file line by line
while IFS= read -r line; do
        # Increment the counter
        ((counter++))

        # Check if the current row is even (every second row)
        if [ $((counter % 2)) -eq 0 ]; then
                continue
        fi

        # Write the line to the output file
        echo "$line" >> "$output_file"

done < "$input_file"
echo "Every second row deleted. Output saved to $output_file"