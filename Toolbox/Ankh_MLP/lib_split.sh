#!/bin/bash

usage() {
  echo "Usage: $0 -i <input_dir>"
  echo "  -i   Directory of the input lib"
  exit 1
}

while getopts "i:h" opt; do
  case $opt in
    i) input_dir="$OPTARG" ;;
    h|\?) usage ;;
  esac
done

# Name of the original TSV file
input_file="${input_dir}/db_lib.tsv"
output_dir="${input_dir}/db_lib"
# Create output directory for storage of future prediction results
output_pre="${input_dir}/double" 

# Check if the directory exists
if [ -d "$output_dir" ]; then
    echo "Directory exists. Continuing the workflow..."
else
    echo "Directory does not exist. Creating the directory..."
    mkdir -p "$output_dir"
    mkdir -p "$output_pre"
fi


# Number of data rows per split file
rows_per_split=5000

# Get the header row
header=$(head -n 1 $input_file)

# Split the file into chunks, excluding the header
tail -n +2 $input_file | split -l $rows_per_split -d - "tem_lib_"

counter=1

# Loop through the generated files and add the header
for file in tem_lib_*
do
    sed -i "1i$header" $file
    mv "$file" "${output_dir}/${counter}.tsv"
    counter=$((counter + 1))
done

echo "spliting done."
