date=2020-11-01
echo $date
echo 'update CSSE data...'
#cd /home/zhgao/COVID19/COVID-19
#git pull
echo 'update features...'
#cd /home/zhgao/COVID-Research/src
#python utils.py

CUDA_VISIBLE_DEVICES=0 nohup sh forecast.epiweek.nbeats.sh $date deaths 7 > deaths.nbeats.7.log&
CUDA_VISIBLE_DEVICES=0 nohup sh forecast.epiweek.nbeats.sh $date deaths 14 > deaths.nbeats.14.log&
CUDA_VISIBLE_DEVICES=1 nohup sh forecast.epiweek.nbeats.sh $date deaths 21 > deaths.nbeats.21.log&
CUDA_VISIBLE_DEVICES=1 nohup sh forecast.epiweek.nbeats.sh $date deaths 28 > deaths.nbeats.28.log&

CUDA_VISIBLE_DEVICES=2 nohup sh forecast.epiweek.nbeats.sh $date confirmed 7 > confirmed.nbeats.7.log&
CUDA_VISIBLE_DEVICES=2 nohup sh forecast.epiweek.nbeats.sh $date confirmed 14 > confirmed.nbeats.14.log&
CUDA_VISIBLE_DEVICES=3 nohup sh forecast.epiweek.nbeats.sh $date confirmed 21 > confirmed.nbeats.21.log&
CUDA_VISIBLE_DEVICES=3 nohup sh forecast.epiweek.nbeats.sh $date confirmed 28 > confirmed.nbeats.28.log&
