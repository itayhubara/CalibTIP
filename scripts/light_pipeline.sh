export model=${1:-"resnet"}
export model_vis=${2:-"resnet50"}

sh scripts/adaquant.sh $model $model_vis 4 4 false
sh scripts/adaquant.sh $model $model_vis 8 8 false
sh scripts/integer-programing.sh $model $model_vis 4 4 8 8 50 loss false

# Uncomment to run first configuration only
#for cfg_idx in 0
for cfg_idx in 0 1 2 3 4 5 6 7 8 9 10 11
do
   # TODO: run bn tuning in loop on all configurations
   echo "Running configuration $cfg_idx"
   sh scripts/bn_tuning.sh resnet resnet50 8 8 $cfg_idx
done
