#!/bin/bash
# sh scripts/ood/odin/mstar_test_osr_odin.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
    --config configs/datasets/mstar/mstar3_seed1.yml \
    configs/datasets/mstar/mstar3_seed1_osr.yml \
    configs/networks/resnet18_64x64.yml \
    configs/pipelines/test/test_osr.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/odin.yml \
    --num_workers 8 \
    --network.checkpoint 'results/mstar3_seed1_resnet18_64x64_base_e100_lr0.01_default/s0/best.ckpt' \
    --mark 0
