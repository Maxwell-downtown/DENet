#!/bin/bash

# Name of the original TSV file
input_file="./GFP_AEQVI/L2-t100.tsv"
output_dir="./GFP_AEQVI/L2_t100"

# Check if the directory exists
if [ -d "$output_dir" ]; then
    echo "Directory exists. Continuing the workflow..."
else
    echo "Directory does not exist. Creating the directory..."
    mkdir -p "$output_dir"
fi

# Number of data rows per split file
rows_per_split=10000

# Get the header row
header=$(head -n 1 $input_file)

# Split the file into chunks, excluding the header
tail -n +2 $input_file | split -l $rows_per_split -d - "tem_lib_"

counter=1

# Loop through the generated files and add the header
for file in tem_lib_*
do
    sed -i "1i$header" $file
    mv "$file" "./GFP_AEQVI/L2_t100/${counter}.tsv"
    counter=$((counter + 1))
done

echo "spliting done."
