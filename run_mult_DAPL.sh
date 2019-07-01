#!/bin/bash

for  i in 10 20 30 40 50 60 70 80 90 
do
	echo "---------------------${i}-----------------------"
	bash run_DAPL.sh $i all $i
	bash run_result_analysis.sh $i all $i
	echo "---------------------------------------------"
	
done
