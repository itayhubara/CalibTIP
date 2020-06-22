export SQUAD_DIR=/media/drive/Datasets/squad
CUDA_VISIBLE_DEVICES=2 python ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --train_file $SQUAD_DIR/train-v1.1.json \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./examples/models/bert_base_uncased_finetuned_squad_base_384 \
    --per_gpu_eval_batch_size=12   \
    --per_gpu_train_batch_size=12   \
    --do_lower_case \
    --local_rank=-1 \

