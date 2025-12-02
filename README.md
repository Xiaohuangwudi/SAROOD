This is the code for paper 'A Comprehensive Framework for Out-of-Distribution Detection and Open-Set Recognition in SAR Targets'.

The codebase is developed using ood_coverage https://github.com/BierOne/ood_coverage.git.

First step:
sh scripts/ood/vos/mstar_train_vos.sh

Second step:
sh scripts/ood/nac/mstar_test_ood_nac.sh 128 1000 "avgpool layer3 layer2 layer1"
