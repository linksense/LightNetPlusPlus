#!/usr/bin/env bash

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

log_dir=/home/huijun/TrainLogs/logs/SharpNet-2018-11-19-17-43-24
tb_dir=/home/huijun/anaconda3/lib/python3.6/site-packages/tensorboard/
python ${tb_dir}main.py --logdir=${log_dir} --port=8008