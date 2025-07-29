#!/bin/bash

usage() {
  echo "Usage: $0 -r <random_seed>"
  exit 1
}

while getopts "r:h" opt; do
  case $opt in
    r) RD="$OPTARG" ;;
    h|\?) usage ;;
  esac
done

input_dir="./GB1/MLDE/${RD}_kfold_t50/epi_lib"
output_file="./GB1/MLDE/${RD}_kfold_t50/R6/epi_double.tsv"
start=1
end=54

header_written=0

for i in $(seq $start $end);
do
    file="$input_dir"/${i}/prediction.tsv
    if [ $header_written == 0 ]; then
        echo head_written=0
        head -n 1 "$file" > "$output_file"
        header_written=1
        tail -n +2 "$file" >> "$output_file"
    else
        tail -n +2 "$file" >> "$output_file"
    fi
done
