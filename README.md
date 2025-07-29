# DENet
Code and data for running DENet.

## Installation ##
After downloading the DENet data files, one can directly configure the conda environment with the following command:
```
conda env create -f DENet_environment.yml
conda activate DENet
```

Note: The Ankh (large) model will be automatically fetched from the Hugging Face Hub on first run. The checkpoint is ~7.5 GB and may take several minutes to download.

The co-mutation information files used by DENet can be found in this [Figshare link](https://figshare.com/s/2224ffd3d20231ea8a45), under the `./Data/Protein_info/seq/` directory. Download the corresponding `***.braw` and add them to the same `./Data/Protein_info/seq/` directory in your DENet file. 

## Running DENet ##
Model training can be done with the following command (take MEK1 for example):
```
python starter.py --train ../Data/Protein_Info/score/MEK1_DE.tsv --fasta ../Data/Protein_Info/seq/MEK1.fasta --comutation ../Data/Protein_Info/seq/MEK1_DEh8k.braw --structure ../Data/Protein_Info/struct/MEK1_wt/ --output_dir ./output/MEK1_DE --epochs 500 --use_gcn --save_log --save_prediction --save_checkpoint
```
The upper training process used co-mutation information extracted from DE experiment trajectories (`MEK1_DEh8k.braw`), you can compare the training results with co-mutation data from traditional MSA of homologous sequences or using no co-mutation at all (change `--comutation ../Data/Protein_Info/seq/MEK1_DEh8k.braw` to `--comutation ../Data/Protein_Info/seq/MEK1.braw` or `--no_comutation`, respectively)

You can also applied the trained model to more testing data with the following command:
```
python starter.py --test ../Data/Protein_Info/lib/MEK1-sin_lib.tsv --fasta ../Data/Protein_Info/seq/MEK1.fasta --comutation ../Data/Protein_Info/seq/MEK1_DEh8k.braw --structure ../Data/Protein_Info/struct/MEK1_wt/ --output_dir ./output/MEK1_DE/sin_mutants --epochs 500 --use_gcn --save_log --save_prediction --saved_model_dir ./output/MEK1_DE/
```
- To apply DENet on your own data, structure and co-mutation information need to be prepared. Though you can skip either of these two extra modality through `--no_comutation`, or without adding `--use_gcn`. 
- For co-mutation information preparation, highly enriched mutant library can be extracted from counts files of different time points along the DE trajectories with `./Seqlib_builder.py` in `./Toolbox/`. This will give you a `.psc` file, which needs to be further processed by CCMpred to generate the eventual `.braw` file. (For DMS data, `.psc` file can be obtained through homologous sequence searching and MSA of the target protein using tools such as hhblits)
- For structure information preparation, once the `.pdb` file is ready, you can directly process the `.pdf` file with `./StructMap.py` in `./Toolbox/`, which will generate the structure file needed for model training. 

