# DENet
Code and data for running DENet.

## Installation ##
After downloading the DENet repository, one can directly configure the conda environment with the following command:
```
conda env create -f DENet_environment.yml
conda activate DENet
```

Note: The Ankh (large) model will be automatically fetched from the Hugging Face Hub on first run. The checkpoint is ~7.5 GB and may take several minutes to download.

The co-mutation information files used by DENet can be found in this [Figshare link](https://figshare.com/s/2224ffd3d20231ea8a45), under the `./Data/Protein_info/seq/` directory. Download the corresponding `***.braw` and add them to the same `./Data/Protein_info/seq/` directory in your DENet file. 

## Running DENet ##
Model training can be done using codes under `./DENet` with the following command (take MEK1 for example):
```
python starter.py --train ../Data/Protein_Info/score/MEK1_DE.tsv \
--fasta ../Data/Protein_Info/seq/MEK1.fasta \
--comutation ../Data/Protein_Info/seq/MEK1_DEh8k.braw \
--structure ../Data/Protein_Info/struct/MEK1_wt/ \
--output_dir ./output/MEK1_DE --epochs 500 --use_gcn \
--save_log --save_prediction --save_checkpoint
```
The upper training process used co-mutation information extracted from DE experiment trajectories (`MEK1_DEh8k.braw`), you can compare the training results with co-mutation data from traditional MSA of homologous sequences or using no co-mutation at all (change `--comutation ../Data/Protein_Info/seq/MEK1_DEh8k.braw` to `--comutation ../Data/Protein_Info/seq/MEK1.braw` or `--no_comutation`, respectively)

You can also applied the trained model to more testing data with the following command:
```
python starter.py --test ../Data/Protein_Info/lib/MEK1-sin_lib.tsv \
--fasta ../Data/Protein_Info/seq/MEK1.fasta \
--comutation ../Data/Protein_Info/seq/MEK1_DEh8k.braw \
--structure ../Data/Protein_Info/struct/MEK1_wt/ \
--output_dir ./output/MEK1_DE/sin_mutants --epochs 500 --use_gcn \
--save_log --save_prediction --saved_model_dir ./output/MEK1_DE/
```
- To apply DENet on your own data, structure and co-mutation information need to be prepared. Though you can skip either of these two extra modality through `--no_comutation`, or without adding `--use_gcn`. 
- For co-mutation information preparation, highly enriched mutant library can be extracted from counts files of different time points along the DE trajectories with `./Seqlib_builder.py` in `./Toolbox/`. This will give you a `.psc` file, which needs to be further processed by CCMpred to generate the eventual `.braw` file. (For DMS data, `.psc` file can be obtained through homologous sequence searching and MSA of the target protein using tools such as hhblits)
- For structure information preparation, once the `.pdb` file is ready, you can directly process the `.pdf` file with `./StructMap.py` in `./Toolbox/`, which will generate the structure file needed for model training.

To perform data collection simulations on GB1, use `simulation.py` in `./GB1_simulation/DE/` for DE method, use `GB1_MLDEtrain.sh` in `./Toolbox/Ankh-MLP/` for the MLDE method. DMS method can be simulated by directly drawing all the single mutants plus 1,000 random double mutants from `./Data/Protein_Info/score/GB1_56aa_all.tsv`
The prediction performance can be compared directly.
For DE-based methods, training and testing could be done with:
```
python starter.py --train ../Data/Protein_Info/score/GB1_DE_2k.tsv \
--fasta ../Data/Protein_Info/seq/GB1.fasta \
--comutation ../Data/Protein_Info/seq/GB1_DEh6k.braw \
--structure ../Data/Protein_Info/struct/GB1_wt/ \
--output_dir ./output/GB1_DE --epochs 500 \
--use_gcn --save_log --save_prediction --save_checkpoint

python starter.py --test ../GB1_simulation/test/set1.tsv \
--fasta ../Data/Protein_Info/seq/GB1.fasta \
--comutation ../Data/Protein_Info/seq/GB1_DEh6k.braw \
--structure ../Data/Protein_Info/struct/GB1_wt/ \
--output_dir ./output/GB1_DE/test/set1 \
--use_gcn --save_log --save_prediction \
--saved_model_dir ./output/GB1_DE/
```
For DMS-based methods, training and testing could be done with:
```
python starter.py --train ../Data/Protein_Info/score/GB1_DMS_2k.tsv \
--fasta ../Data/Protein_Info/seq/GB1.fasta \
--comutation ../Data/Protein_Info/seq/GB1.braw \
--structure ../Data/Protein_Info/struct/GB1_wt/ \
--output_dir ./output/GB1_DMS --epochs 500 \
--use_gcn --save_log --save_prediction --save_checkpoint

python starter.py --test ../GB1_simulation/test/set1.tsv \
--fasta ../Data/Protein_Info/seq/GB1.fasta \
--comutation ../Data/Protein_Info/seq/GB1.braw \
--structure ../Data/Protein_Info/struct/GB1_wt/ \
--output_dir ./output/GB1_DMS/test/set1 --use_gcn \
--save_log --save_prediction --saved_model_dir ./output/GB1_DMS/
```
For MLDE-based methods, prediction results are directly shown in the output after running `GB1_MLDEtrain.sh`(under directory `./Toolbox/Ankh_MLP/`)

## DENet with in-silico directed evolution ##
In-silico DE can serve as an alternative sources for co-mutation information extraction and used for DENet training.
To perform in silico directed evolution with a simplified Ankh-MLP model, use the model in `./Toolbox` (go to  `./Toolbox/Ankh_MLP`). You can train the model similarly with the following codes(take GFP_AEQVI for example):
```
python starter.py --train ../../Data/Protein_Info/score/GFP_AEQVI/sin.tsv \
--fasta ../../Data/Protein_Info/seq/GFP_AEQVI.fasta \
--output_dir output/sin_high/GFP_AEQVI --n_ensembles 3 \
--save_log --save_prediction --save_checkpoint

python starter.py --test ../../Data/Protein_Info/lib/GFP_AEQVI-sin_lib.tsv \
--fasta ../../Data/Protein_Info/seq/GFP_AEQVI.fasta \
--output_dir output/sin_high/GFP_AEQVI/sin/ --n_ensembles 3 \
--save_log --save_prediction --saved_model_dir output/sin_high/GFP_AEQVI
```
After predicting the scores for all single mutants for GFP_AEQVI(`GFP_AEQVI-sin_lib.tsv`), the top 100 scoring single mutants are selected and used to compose the double mutant library for virtual screening with `./library/db_mut.py` under Ankh_MLP. To avoid CUDA out of memory errors, use `./library/lib_split.sh` to split the screening file into smaller pieces. 
Testing on the screening files can then be done with `batch_test.sh` under the `./Ankh_MLP` directory.
The screening results can then be collected with `lib_collect.sh` (`./output/lib_collect.sh`).
Top scoring mutants are selected into a separate tsv file(`GFP_AEQVI_t10k.tsv`), which were processed by `Seqlib_builder2.py` in `./Toolbox/` into `.psc` file. The `.psc` file can be further processed by CCMpred to generate the `.braw` file we need(`GFP_AEQVI_DEh10k.braw`).
The full DENet can be trained and tested using the co-mutation data from the `.braw file` we obtained above:
```
python starter.py --train ../Data/Protein_Info/score/GFP_AEQVI/sin.tsv \
--fasta ../Data/Protein_Info/seq/GFP_AEQVI.fasta \
--comutation ../Data/Protein_Info/seq/GFP_AEQVI_DEh10k.braw \
--structure ../Data/Protein_Info/struct/GFP_AEQVI/ \
--output_dir ./output/GFP_AEQVI --epochs 500 \
--use_gcn --save_log --save_prediction --save_checkpoint

python starter.py --test ../Data/Protein_Info/score/GFP_AEQVI/double.tsv \
--fasta ../Data/Protein_Info/seq/GFP_AEQVI.fasta \
--comutation ../Data/Protein_Info/seq/GFP_AEQVI_DEh10k.braw \
--structure ../Data/Protein_Info/struct/GFP_AEQVI/ \
--output_dir ./output/GFP_AEQVI/doubleTest --use_gcn \
--save_log --save_prediction --saved_model_dir ./output/GFP_AEQVI/
```
The upper process can also be compared with DENet using co-mutation information from MSA of homologous sequences or without co-mutation information by changing `--comutation ../Data/Protein_Info/seq/GFP_AEQVI_DEh10k.braw` to `--comutation ../Data/Protein_Info/seq/GFP_AEQVI.braw` or `--no_comutation`

