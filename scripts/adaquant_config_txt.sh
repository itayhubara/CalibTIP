export datasets_dir=/home/Datasets
export model=${1:-"resnet"}
export model_vis=${2:-"resnet50"}
export nbits_weight=${3:-4}
export nbits_act=${4:-4}
export num_samples=${6:-''}
export configs_file=${7:-''}
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
python main_ft.py --model $model --save $workdir -b 128  -lfv $model_vis --model-config "{'batch_norm': False}"
for cr in 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24
do
#python create_calib_folder.py $num_samples
# measure range and zero point on calibset
prec_dict=$(python ip_txt_config_parser.py --config-file $configs_file --compression_rate $cr)
python main_ft_adapt.py --model $model  --nbits_weight $nbits_weight --nbits_act $nbits_act --num-sp-layers $num_sp_layers --evaluate results/$workdir/$model.absorb_bn --model-config "{'batch_norm': False,'measure': True, 'perC': $perC}" -b 100 --rec --dataset imagenet_calib_$num_samples --datasets-dir $datasets_dir --device-ids 0 -lpd "$prec_dict"
if [ "$5" = True ]; then
# Run adaquant to minimize MSE of the output with respect to range, zero point and small perturations in parameters
python main_ft_adapt.py --optimize-weights  --nbits_weight $nbits_weight --nbits_act $nbits_act  --num-sp-layers $num_sp_layers  --model $model -b 200 --evaluate results/$workdir/$model.absorb_bn.measure$perC_suffix --model-config "{'batch_norm': False,'measure': False, 'perC': $perC}" --dataset imagenet_calib_$num_samples --datasets-dir $datasets_dir --adaquant  --device-ids 0 -lpd "$prec_dict" --compression_rate $cr
fi
done
