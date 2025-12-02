#!/bin/bash
# sh scripts/ood/ebo/mstar_test_ood_ebo.sh
# GPU=1
# CPU=1
# node=36
# jobname=openood
# PYTHONPATH='.':$PYTHONPATH \
# python main.py \
#     --config configs/datasets/mstar/mstar.yml \
#             configs/datasets/mstar/mstar_ood.yml \
#             configs/networks/resnet18_64x64.yml \
#             configs/pipelines/test/test_ood.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/msp.yml \
#     --num_workers 8 \
#     --network.checkpoint './results/mstar_resnet18_64x64_base_e100_lr0.1_default/s0/best.ckpt' \
#     --mark 0 \
#     --merge_option merge

# 使用新脚本进行评估
python scripts/eval_ood.py \
    --id-data mstar \
    --root ./results/mstar_resnet18_64x64_base_e100_lr0.01_default \
    --postprocessor ebo \
    --save-score --save-csv
