date=2020-12-06
echo $date
echo 'update features...'
#cd /home/zhgao/COVID-Research/src
python utils.py

CUDA_VISIBLE_DEVICES=0 nohup sh forecast.epiweek.sh $date deaths 7 > deaths.7.log&
CUDA_VISIBLE_DEVICES=0 nohup sh forecast.epiweek.sh $date deaths 14 > deaths.14.log&
CUDA_VISIBLE_DEVICES=1 nohup sh forecast.epiweek.nbeats.sh $date deaths 7 > deaths.nbeats.7.log&
CUDA_VISIBLE_DEVICES=1 nohup sh forecast.epiweek.nbeats.sh $date deaths 14 > deaths.nbeats.14.log&


CUDA_VISIBLE_DEVICES=2 nohup sh forecast.epiweek.sh $date confirmed 7 > confirmed.7.log&
CUDA_VISIBLE_DEVICES=2 nohup sh forecast.epiweek.sh $date confirmed 14 > confirmed.14.log&
CUDA_VISIBLE_DEVICES=3 nohup sh forecast.epiweek.nbeats.sh $date confirmed 7 > confirmed.nbeats.7.log&
CUDA_VISIBLE_DEVICES=3 nohup sh forecast.epiweek.nbeats.sh $date confirmed 14 > confirmed.nbeats.14.log&

