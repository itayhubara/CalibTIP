export datasets_dir=/media/drive/Datasets
export model=${1:-"resnet"}
export model_vis=${2:-"resnet50"}
export nbits_weight=${3:-4}
export nbits_act=${4:-4}
export perC=True
export num_sp_layers=-1
export perC_suffix=''
export adaquant=True
if [ "$adaquant" = True ]; then
    export adaquant_suffix='.adaquant'
fi
export perC_suffix=''
if [ "$perC" = True ]; then
    export perC_suffix='_perC'
fi
export workdir=${model_vis}_w$nbits_weight'a'$nbits_act$adaquant_suffix

cfg_idx=${5:-0}
prec_dict=$(python ip_config_parser.py --cfg-idx $cfg_idx --config-file results/$workdir/IP_${model_vis}_loss.txt --column Configuration)
ckp=$(python ip_config_parser.py --cfg-idx $cfg_idx --config-file results/$workdir/IP_${model_vis}_loss.txt --column state_dict_path)

# Run bn tuning
python main.py -lpd "$prec_dict" --batch-norn-tuning --model $model -lfv $model_vis -b 200 --evaluate $ckp --model-config "{'batch_norm': False,'measure': False, 'perC': $perC}" --dataset imagenet_calib --datasets-dir $datasets_dir

