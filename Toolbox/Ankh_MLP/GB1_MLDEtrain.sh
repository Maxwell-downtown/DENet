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

RD=${RD:-42}

echo "Using random seed: $RD"

python starter_kfold.py --train ../../GB1_simulation/MLDE/R1/GB1_round1_t50.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R1 --save_log --save_prediction --save_checkpoint --random ${RD}

python starter_kfold.py --test ../../Data/Protein_Info/score/GB1_56aa_single.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R1/single --save_log --save_prediction --random ${RD} --saved_model_dir output/GB1/MLDE/${RD}_kfold_t50/R1/kfold/

python SpearmanC.py -i output/GB1/MLDE/${RD}_kfold_t50/R1/single/prediction.tsv

cp output/GB1/MLDE/${RD}_kfold_t50/R1/single/prediction.tsv output/GB1/MLDE/${RD}_kfold_t50/R1/${RD}_R1_single.tsv

mkdir ../../GB1_simulation/MLDE/R2/

python call_score.py -i1 ./output/GB1/MLDE/${RD}_kfold_t50/R1/${RD}_R1_single.tsv -i2 ../../GB1_simulation/MLDE/R1/GB1_round1_t50.tsv -o ../../GB1_simulation/MLDE/R2/${RD}_kfold_t50_r2.tsv

python starter_kfold.py --train ../../GB1_simulation/MLDE/R2/${RD}_kfold_t50_r2.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R2 --save_log --save_prediction --save_checkpoint --random ${RD} --resume_dir output/GB1/MLDE/${RD}_kfold_t50/R1/kfold/

python starter_kfold.py --test ../../Data/Protein_Info/score/GB1_56aa_single.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R2/single --save_log --save_prediction --random ${RD} --saved_model_dir output/GB1/MLDE/${RD}_kfold_t50/R2/kfold 

python SpearmanC.py -i output/GB1/MLDE/${RD}_kfold_t50/R2/single/prediction.tsv

cp output/GB1/MLDE/${RD}_kfold_t50/R2/single/prediction.tsv output/GB1/MLDE/${RD}_kfold_t50/R2/${RD}_R2_single.tsv

mkdir ../../GB1_simulation/MLDE/R3/

python call_score.py -i1 ./output/GB1/MLDE/${RD}_kfold_t50/R2/${RD}_R2_single.tsv -i2 ../../GB1_simulation/MLDE/R2/${RD}_kfold_t50_r2.tsv -o ../../GB1_simulation/MLDE/R3/${RD}_kfold_t50_r3.tsv

python starter_kfold.py --train ../../GB1_simulation/MLDE/R3/${RD}_kfold_t50_r3.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R3 --save_log --save_prediction --save_checkpoint --random ${RD} --resume_dir output/GB1/MLDE/${RD}_kfold_t50/R2/kfold/

python starter_kfold.py --test ../../Data/Protein_Info/score/GB1_56aa_single.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R3/single --save_log --save_prediction --random ${RD} --saved_model_dir output/GB1/MLDE/${RD}_kfold_t50/R3/kfold

python SpearmanC.py -i output/GB1/MLDE/${RD}_kfold_t50/R3/single/prediction.tsv

cp output/GB1/MLDE/${RD}_kfold_t50/R3/single/prediction.tsv output/GB1/MLDE/${RD}_kfold_t50/R3/${RD}_R3_single.tsv

mkdir ../../GB1_simulation/MLDE/R4/

python call_score.py -i1 ./output/GB1/MLDE/${RD}_kfold_t50/R3/${RD}_R3_single.tsv -i2 ../../GB1_simulation/MLDE/R3/${RD}_kfold_t50_r3.tsv -o ../../GB1_simulation/MLDE/R4/${RD}_kfold_t50_r4.tsv

python starter_kfold.py --train ../../GB1_simulation/MLDE/R4/${RD}_kfold_t50_r4.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R4 --save_log --save_prediction --save_checkpoint --random ${RD} --resume_dir output/GB1/MLDE/${RD}_kfold_t50/R3/kfold/

python starter_kfold.py --test ../../Data/Protein_Info/score/GB1_56aa_single.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R4/single --save_log --save_prediction --random ${RD} --saved_model_dir output/GB1/MLDE/${RD}_kfold_t50/R4/kfold

python SpearmanC.py -i output/GB1/MLDE/${RD}_kfold_t50/R4/single/prediction.tsv

cp output/GB1/MLDE/${RD}_kfold_t50/R4/single/prediction.tsv output/GB1/MLDE/${RD}_kfold_t50/R4/${RD}_R4_single.tsv

python db_lib_genarator.py -i output/GB1/MLDE/${RD}_kfold_t50/R4/${RD}_R4_single.tsv -o output/GB1/MLDE/${RD}_kfold_t50/R4/db_lib.tsv

bash lib_split.sh -i output/GB1/MLDE/${RD}_kfold_t50/R4

bash GB1_dbtest.sh -r 4 -s ${RD}

mkdir ../../GB1_simulation/MLDE/R5/

python call_score2.py -i1 ./output/GB1/MLDE/${RD}_kfold_t50/R4/${RD}_R4_double.tsv -i2 ../../GB1_simulation/MLDE/R4/${RD}_kfold_t50_r4.tsv -r0 ../../Data/Protein_Info/score/GB1_56aa_double.tsv -o ../../GB1_simulation/MLDE/R5/${RD}_kfold_t50_r5.tsv

python starter_kfold.py --train ../../GB1_simulation/MLDE/R5/${RD}_kfold_t50_r5.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R5 --save_log --save_prediction --save_checkpoint --random ${RD} --resume_dir output/GB1/MLDE/${RD}_kfold_t50/R4/kfold/

python starter_kfold.py --test ../../Data/Protein_Info/score/GB1_56aa_single.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R5/single --save_log --save_prediction --random ${RD} --saved_model_dir output/GB1/MLDE/${RD}_kfold_t50/R5/kfold

python SpearmanC.py -i output/GB1/MLDE/${RD}_kfold_t50/R5/single/prediction.tsv

cp output/GB1/MLDE/${RD}_kfold_t50/R5/single/prediction.tsv output/GB1/MLDE/${RD}_kfold_t50/R5/${RD}_R5_single.tsv

python db_lib_genarator.py -i output/GB1/MLDE/${RD}_kfold_t50/R5/${RD}_R5_single.tsv -o output/GB1/MLDE/${RD}_kfold_t50/R5/db_lib.tsv

bash lib_split.sh -i output/GB1/MLDE/${RD}_kfold_t50/R5

bash GB1_dbtest.sh -r 5 -s ${RD}

mkdir ../../GB1_simulation/MLDE/R6/

python call_score2.py -i1 ./output/GB1/MLDE/${RD}_kfold_t50/R5/${RD}_R5_double.tsv -i2 ../../GB1_simulation/MLDE/R5/${RD}_kfold_t50_r5.tsv -r0 ../../Data/Protein_Info/score/GB1_56aa_double.tsv -o ../../GB1_simulation/MLDE/R6/${RD}_kfold_t50_r6.tsv

python starter_kfold.py --train ../../GB1_simulation/MLDE/R6/${RD}_kfold_t50_r6.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R6 --save_log --save_prediction --save_checkpoint --random ${RD} --resume_dir output/GB1/MLDE/${RD}_kfold_t50/R5/kfold/

python starter_kfold.py --test ../../data/DENet/GB1/score/GB1_56aa_single.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R6/single --save_log --save_prediction --random ${RD} --saved_model_dir output/GB1/MLDE/${RD}_kfold_t50/R6/kfold

python SpearmanC.py -i output/GB1/MLDE/${RD}_kfold_t50/R6/single/prediction.tsv

python starter_kfold.py --test ../../GB1_simulation/test/set1.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R6/test26/set1 --save_log --save_prediction --random ${RD} --saved_model_dir output/GB1/MLDE/${RD}_kfold_t50/R6/kfold

python SpearmanC.py -i output/GB1/MLDE/${RD}_kfold_t50/R6/test26/set1/prediction.tsv

python starter_kfold.py --test ../../GB1_simulation/test/set2.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R6/test26/set2 --save_log --save_prediction --random ${RD} --saved_model_dir output/GB1/MLDE/${RD}_kfold_t50/R6/kfold

python SpearmanC.py -i output/GB1/MLDE/${RD}_kfold_t50/R6/test26/set2/prediction.tsv

python starter_kfold.py --test ../../GB1_simulation/test/set3.tsv --fasta ../../Data/Protein_Info/seq/GB1.fasta --output_dir output/GB1/MLDE/${RD}_kfold_t50/R6/test26/set3 --save_log --save_prediction --random ${RD} --saved_model_dir output/GB1/MLDE/${RD}_kfold_t50/R6/kfold

python SpearmanC.py -i output/GB1/MLDE/${RD}_kfold_t50/R6/test26/set3/prediction.tsv
