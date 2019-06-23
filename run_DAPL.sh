#!/bin/bash

exp_no=$1

exp_dir="Experiment$1"

if [ ! -d "$exp_dir" ]; then
	mkdir $exp_dir
	echo "$exp_dir" >> .gitignore
fi

cd Experiment
python3 preprocess_BioDataset_new1.py 100 Experiment$1 1200
cd ..

results_dir="results"

if [ ! -d "$exp_dir/$results_dir" ]; then
        mkdir $exp_dir/$results_dir
fi


python3 DAPL_base.py --input_file Experiment$1/BioDataset1 --shape 0,1200 --save_results True --output_filePath Experiment$1/$results_dir



python3 result_analysis.py --dataset Experiment$1/BioDataset1/rating.npz --recons results.npy --mask mask.npy --result_file_path Experiment$1/$results_dir/output_results.csv 


