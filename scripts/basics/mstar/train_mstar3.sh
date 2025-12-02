#!/bin/bash
# sh scripts/basics/mstar/train_mstar3.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
    --config configs/datasets/mstar/mstar3_seed1.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_64x64.yml \
    configs/pipelines/train/baseline.yml \
    --seed 0
