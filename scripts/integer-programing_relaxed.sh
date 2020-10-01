export datasets_dir=/home/Datasets
export model=${1:-"resnet"}
export model_vis=${2:-"resnet50"}
export nbits_switch="2 3 4 5 6 7 8"
export nbits_weight_m2=${3:-8}
export nbits_act_m2=${4:-8}
export perC=True
export adaquant=True
export precisions="2;3;4;5;6;7;8"
export min_compression=0.13
export max_compression=0.25

for nb in $nbits_switch; do
    export nbits_weight_m1=$nb
    export nbits_act_m1=$nb
    export adaquant_suffix=''
    export do_not_use_adaquant=--do_not_use_adaquant
    if [ "$7" = True ]; then
        export adaquant_suffix='.adaquant'
        export do_not_use_adaquant=''
    fi
    export perC_suffix=''
    if [ "$perC" = True ]; then
        export perC_suffix='_perC'
    fi
    export workdir_m1=${model_vis}_w$nbits_weight_m1'a'$nbits_act_m1$adaquant_suffix
    export workdir_m2=${model_vis}_w$nbits_weight_m2'a'$nbits_act_m2$adaquant_suffix
    
    export num_sp_layers=-1
    export depth=${5:-50}
    export loss=${6:-'loss'}
    export layer_by_layer=$layer_by_layer;results/$workdir_m2/$model.absorb_bn.measure$perC_suffix$adaquant_suffix.per_layer_accuracy.A$nbits_weight_m1.W$nbits_weight_m1.csv
    
    #Extract per layer loss delta
    python main.py --model $model --evaluate results/$workdir_m2/$model.absorb_bn'.measure'$perC_suffix$adaquant_suffix --model-config "{'batch_norm': False,'measure': False, 'perC': $perC, 'depth': $depth}" -b 100 --dataset imagenet_calib --datasets-dir $datasets_dir --int8_opt_model_path results/$workdir_m2/$model.absorb_bn'.measure'$perC_suffix$adaquant_suffix --int4_opt_model_path results/$workdir_m1/$model.absorb_bn'.measure'$perC_suffix$adaquant_suffix --names-sp-layers '' --per-layer
done    
#Run IP algorithm to obtain best topology
python mpip_compression_pytorch_multi.py --model $model  --model_vis $model_vis --ip_method $loss --precisions $precisions --layer_by_layer_files $layer_by_layer --min_compression $min_compression --max_compression $max_compression $do_not_use_adaquant --datasets-dir $datasets_dir 
