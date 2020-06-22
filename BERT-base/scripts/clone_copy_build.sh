git clone https://github.com/huggingface/transformers
cp src/layer_sensativity.py transformers/
cp src/run_squad_adaquant.py transformers/examples/question-answering/
mv transformers/src/transformers/modeling_bert.py transformers/src/transformers/modeling_bert_fp32.py
cp src/modeling_* transformers/src/transformers/
cp src/mse_optimization.py transformers/src/transformers/
cd transformers
pip install .


