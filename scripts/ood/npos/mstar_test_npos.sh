#!/bin/bash
# sh scripts/ood/npos/mstar_test_npos.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/eval_ood.py \
   --id-data mstar3 \
   --root ./results/mstar3_seed1_npos_net_npos_e100_lr0.1_default \
   --postprocessor npos \
   --save-score --save-csv
