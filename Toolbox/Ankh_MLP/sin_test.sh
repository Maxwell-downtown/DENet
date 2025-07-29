#!/bin/bash
#SBATCH -J 0929
#SBATCH -o 0929_test.out
#SBATCH -p gpu_l40
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --gres=gpu:1

python starter.py --test ../../data/sin_high_com/mutation_lib/0929-sin_lib.tsv --fasta ../../data/sin_high_com/seq1/0929.fasta --output_dir output/sin_high/0929/sin --n_ensembles 3 --save_log --save_prediction --saved_model_dir ./output/sin_high/0929/
