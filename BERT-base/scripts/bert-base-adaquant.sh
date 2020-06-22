export SQUAD_DIR=/media/drive/Datasets/squad
export FP_MODEL_DIR=/media/data/transformers/ 
export SQUAD_BERT_NBITS=8
# measure
CUDA_VISIBLE_DEVICES=0 python ./examples/question-answering/run_squad_adaquant.py \
    --model_type bert \
    --model_name_or_path $FP_MODEL_DIR/examples/models/bert_base_uncased_finetuned_squad_base_384 \
    --do_eval \
    --calib_file $SQUAD_DIR/calib-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./examples/models/bert-base-measure-perc-w$SQUAD_BERT_NBITS'a'$SQUAD_BERT_NBITS \
    --per_gpu_eval_batch_size=12   \
    --per_gpu_train_batch_size=12   \
    --do_lower_case \
    --local_rank=-1 \
    --quant-config "{'quantize': True, 'measure': True,'num_bits': 4, 'num_bits_weight': 4, 'perC': True, 'cal_qparams': False}" \

# adaquant
CUDA_VISIBLE_DEVICES=0 python ./examples/question-answering/run_squad_adaquant.py \
    --model_type bert \
    --model_name_or_path  ./examples/models/bert-base-measure-perc-w$SQUAD_BERT_NBITS'a'$SQUAD_BERT_NBITS \
    --do_eval \
    --calib_file $SQUAD_DIR/calib-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./examples/models/bert-base-adaquant-perc-w4a4 \
    --per_gpu_eval_batch_size=12   \
    --per_gpu_train_batch_size=12   \
    --do_lower_case \
    --local_rank=-1 \
    --optimize_weights \
    --quant-config "{'quantize': True, 'measure': False,'num_bits': $SQUAD_BERT_NBITS, 'num_bits_weight': $SQUAD_BERT_NBITS, 'perC': True, 'cal_qparams': False}" \
