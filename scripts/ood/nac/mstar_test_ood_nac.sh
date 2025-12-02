#!/bin/bash
# sh scripts/ood/nac/mstar_test_ood_nac.sh 128 1000 "avgpool layer3 layer2 layer1" aps
BATCH=$1
SubNum=$2
LAYER=$3
APS=$4

if [[ "aps" = "$APS" ]]; then
  python scripts/eval_ood.py \
      --id-data mstar \
      --root ./results/mstar_resnet18_64x64_base_e100_lr0.01_default \
      --postprocessor nac \
      --batch-size $BATCH \
      --valid-num $SubNum \
      --layer-names $LAYER \
      --save-score --aps --save-csv
else
  python scripts/eval_ood.py \
      --id-data mstar \
      --root ./results/mstar_resnet18_64x64_vos_e100_lr0.001_default \
      --postprocessor nac \
      --batch-size $BATCH \
      --valid-num $SubNum \
      --layer-names $LAYER \
      --save-score --save-csv
fi
