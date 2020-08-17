CUDA_VISIBLE_DEVICES=0 python main_task.py --model_type sandwich --data_fp ../data/daily_mobility_US_7.csv --exp_dir ../US_sandwich_7_06_07 --forecast_date 2020-06-07 --label deaths_target
CUDA_VISIBLE_DEVICES=0 python main_task.py --model_type sandwich --data_fp ../data/daily_mobility_US_7.csv --exp_dir ../US_sandwich_7_06_14 --forecast_date 2020-06-14 --label deaths_target
CUDA_VISIBLE_DEVICES=0 python main_task.py --model_type sandwich --data_fp ../data/daily_mobility_US_7.csv --exp_dir ../US_sandwich_7_06_21 --forecast_date 2020-06-21 --label deaths_target
CUDA_VISIBLE_DEVICES=0 python main_task.py --model_type sandwich --data_fp ../data/daily_mobility_US_7.csv --exp_dir ../US_sandwich_7_06_28 --forecast_date 2020-06-28 --label deaths_target
