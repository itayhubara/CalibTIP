# BERT-BASE over SQuAD1.1

This repository adds files to huggingface/transformers repo as to enable adaquant optimization.
The repo is tested on Python 3.6+, PyTorch 1.0.0+ (PyTorch 1.3.1+ for examples) and TensorFlow 2.0.
As suggested by [🤗 Transformers](https://github.com/huggingface/transformers) you should install it in a virtual environment.

In your virtual environment please use the follwing script to clone and copy the releven files:
```bash
sh scripts/clone_copy_build.sh
```
To reproduce the results please make sure that you have [SQuAD-v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset and pretrained base model. You can finetune FP32 BERT base on SQuAD using:
```bash
sh ../scripts/bert-base-squad1.1-384.sh
```
Next create calibration dataset from training file and run AdaQuant
```bash
python ../scripts/create_calib_data.py
sh ../scripts/bert-base-adaquant.sh
```

