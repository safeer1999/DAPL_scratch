results_dir='results'



python3 result_analysis.py --dataset Experiment$1/results/train.npy --dataset_type npy --recons Experiment$1/$results_dir/recons.npy --mask Experiment$1/$results_dir/mask.npy --result_file_path Experiment$1/$results_dir/output_results_train.csv 


python3 result_analysis.py --dataset Experiment$1/$results_dir/val.npy --dataset_type npy --recons Experiment$1/$results_dir/val_recons.npy --mask Experiment$1/$results_dir/val_mask.npy --result_file_path Experiment$1/$results_dir/output_results_val.csv


