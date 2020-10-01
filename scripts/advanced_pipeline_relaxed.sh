sh scripts/adaquant.sh resnet resnet50 2 2 True
sh scripts/adaquant.sh resnet resnet50 3 3 True
sh scripts/adaquant.sh resnet resnet50 4 4 True
sh scripts/adaquant.sh resnet resnet50 5 5 True
sh scripts/adaquant.sh resnet resnet50 6 6 True
sh scripts/adaquant.sh resnet resnet50 7 7 True
sh scripts/adaquant.sh resnet resnet50 8 8 True
sh scripts/integer-programing_relaxed.sh resnet resnet50 8 8 50 loss True

# Uncomment to run first configuration only
#for cfg_idx in 0
for cfg_idx in 0 1 2 3 4 5 6 7 8 9 10 11
do
   # TODO: run bn and bias tuning in loop on all configurations
   sh scripts/bn_tuning.sh resnet resnet50 8 8 $cfg_idx
   sh scripts/bias_tuning.sh resnet resnet50 8 8 $cfg_idx
done
