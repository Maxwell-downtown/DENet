#!/bin/bash

input_dir="./sin_high/GFP_AEQVI/L2-t100"
output_file="./sin_high/GFP_AEQVI/L2-t100.tsv"

start=1
end=45

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
