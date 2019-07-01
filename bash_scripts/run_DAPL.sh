#!/bin/bash

exp_no=$1

exp_dir="Experiment$1"

if [ ! -d "$exp_dir" ]; then
	mkdir $exp_dir
	echo "$exp_dir" >> .gitignore
fi

cd Experiment
python3 preprocess_BioDataset_new1.py 100 Experiment$1 $2
cd ..

results_dir="results"

if [ ! -d "$exp_dir/$results_dir" ]; then
        mkdir $exp_dir/$results_dir
fi


python3 train_DAPL.py --input_file Experiment$1/BioDataset1 --save_results True --output_filePath Experiment$1/$results_dir --epochs 1000 --missing_perc $3 --save_model True --model_dir Experiment$1/model/


