#!/bin/bash

date=$1
target=$2
days=$3
for i in $(seq 1 20)
do
python main_task.py --model_type sandwich --data_fp ../data/daily_mobility_US_$days.csv  --exp_dir ../weights_major/US_$target\_$date\_$days\_seed_$i --forecast_date $date --label $target\_target --horizon $days --random_seed $i --use_mobility false --early_stop_epochs 10 --use_saintdataset true --saint_sample_type random_walk
rm -rf ../weights_major/US_$target\_$date\_$days\_seed_$i/Checkpoint
done
