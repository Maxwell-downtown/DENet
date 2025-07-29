#!/bin/bash

usage() {
  echo "Usage: $0 -r <round_number> -s <random_seed>"
  echo "  -r   Number of round"
  echo "  -s   Random seed"
  exit 1
}

while getopts "r:s:h" opt; do
  case $opt in
    r) round="$OPTARG" ;;
    s) seed="$OPTARG" ;;
    h|\?) usage ;;
  esac
done

start=1
end=5

for i in $(seq $start $end); do
    python starter_kfold.py --test ./output/GB1/MLDE/${seed}_kfold_t50/R${round}/db_lib/${i}.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${seed}_kfold_t50/R${round}/double/${i} --save_log --save_prediction --saved_model_dir output/GB1/MLDE/${seed}_kfold_t50/R${round}/kfold/
done

input_dir="./output/GB1/MLDE/${seed}_kfold_t50/R${round}/double/"
output_file="./output/GB1/MLDE/${seed}_kfold_t50/R${round}/${seed}_R${round}_double.tsv"

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

