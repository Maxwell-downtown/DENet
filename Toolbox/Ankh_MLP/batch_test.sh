start=1
end=45

for i in $(seq $start $end); do
	python starter.py --test ./library/GFP_AEQVI/L2_t100/${i}.tsv --fasta ../../Data/Protein_Info/seq/GFP_AEQVI.fasta --output_dir output/sin_high/GFP_AEQVI/L2-t100/${i} --n_ensembles 3 --save_log --save_prediction --saved_model_dir ./output/sin_high/GFP_AEQVI/
	echo "Number of files being processed: ${i}"
done
