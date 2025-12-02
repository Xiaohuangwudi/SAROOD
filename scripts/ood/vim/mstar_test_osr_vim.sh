#!/bin/bash
# sh scripts/ood/vim/mstar_test_osr_vim.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

# python main.py \
#     --config configs/datasets/mstar/mstar3_seed1.yml \
#     configs/datasets/mstar/mstar3_seed1_osr.yml \
#     configs/networks/resnet18_64x64.yml \
#     configs/pipelines/test/test_osr.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/vim.yml \
#     --num_workers 8 \
#     --network.checkpoint 'results/mstar3_seed1_resnet18_64x64_base_e100_lr0.01_default/s0/best.ckpt' \
#     --mark 0 \
#     --postprocessor.postprocessor_args.dim 42


python scripts/eval_ood.py \
   --id-data mstar3 \
   --root ./results/mstar3_seed1_resnet18_64x64_vos_e100_lr0.001_default \
   --postprocessor vim \
   --save-score --save-csv