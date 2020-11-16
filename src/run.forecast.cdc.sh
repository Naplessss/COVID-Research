date=2020-11-08
echo $date
echo 'update CSSE data...'
#cd /home/zhgao/COVID19/COVID-19
#git pull
echo 'update features...'
#cd /home/zhgao/COVID-Research/src
#python utils.py

CUDA_VISIBLE_DEVICES=0 nohup sh forecast.epiweek.sh $date deaths 7 > deaths.7.log&
CUDA_VISIBLE_DEVICES=0 nohup sh forecast.epiweek.sh $date deaths 14 > deaths.14.log&
CUDA_VISIBLE_DEVICES=1 nohup sh forecast.epiweek.sh $date deaths 21 > deaths.21.log&
CUDA_VISIBLE_DEVICES=1 nohup sh forecast.epiweek.sh $date deaths 28 > deaths.28.log&

CUDA_VISIBLE_DEVICES=2 nohup sh forecast.epiweek.sh $date confirmed 7 > confirmed.7.log&
CUDA_VISIBLE_DEVICES=2 nohup sh forecast.epiweek.sh $date confirmed 14 > confirmed.14.log&
CUDA_VISIBLE_DEVICES=3 nohup sh forecast.epiweek.sh $date confirmed 21 > confirmed.21.log&
CUDA_VISIBLE_DEVICES=3 nohup sh forecast.epiweek.sh $date confirmed 28 > confirmed.28.log&
