#!/bin/bash
# sh scripts/ood/vim/mstar_test_ood_vim.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

# python main.py \
#     --config configs/datasets/mstar/mstar.yml \
#     configs/datasets/mstar/mstar_ood.yml \
#     configs/networks/resnet18_64x64.yml \
#     configs/pipelines/test/test_ood.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/vim.yml \
#     --num_workers 8 \
#     --network.checkpoint 'results/mstar_resnet18_64x64_base_e100_lr0.1_default/s0/best.ckpt' \
#     --mark 0 \
#     --postprocessor.postprocessor_args.dim 256

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data mstar \
   --root ./results/mstar_resnet18_64x64_base_e100_lr0.1_default \
   --postprocessor vim \
   --save-score --save-csv
