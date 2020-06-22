export datasets_dir=/media/drive/Datasets
export model=${1:-"resnet"}
export model_vis=${2:-"resnet50"}
export nbits_weight=${3:-4}
export nbits_act=${4:-4}
export adaquant_suffix=''
if [ "$5" = True ]; then
    export adaquant_suffix='.adaquant'
fi
export workdir=${model_vis}_w$nbits_weight'a'$nbits_act$adaquant_suffix
export perC=True
export num_sp_layers=-1
export perC_suffix=''
if [ "$perC" = True ] ; then
export perC_suffix='_perC'
fi
# download and absorb_bn resnet50 and
python main.py --model $model --save $workdir -b 128  -lfv $model_vis --model-config "{'batch_norm': False}"

# measure range and zero point on calibset
python main.py --model $model  --nbits_weight $nbits_weight --nbits_act $nbits_act --num-sp-layers $num_sp_layers --evaluate results/$workdir/$model.absorb_bn --model-config "{'batch_norm': False,'measure': True, 'perC': $perC}" -b 100 --rec --dataset imagenet_calib --datasets-dir $datasets_dir
if [ "$5" = True ]; then
# Run adaquant to minimize MSE of the output with respect to range, zero point and small perturations in parameters
python main.py --optimize-weights  --nbits_weight $nbits_weight --nbits_act $nbits_act  --num-sp-layers $num_sp_layers  --model $model -b 200 --evaluate results/$workdir/$model.absorb_bn.measure$perC_suffix --model-config "{'batch_norm': False,'measure': False, 'perC': $perC}" --dataset imagenet_calib --datasets-dir $datasets_dir --adaquant
fi
