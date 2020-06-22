git clone https://github.com/huggingface/transformers
cd transformers
git checkout 5620033115e571013325d017bcca92991b0a4ace
cd ..
cp src/layer_sensativity.py transformers/
cp src/run_squad_adaquant.py transformers/examples/question-answering/
mv transformers/src/transformers/modeling_bert.py transformers/src/transformers/modeling_bert_fp32.py
cp src/modeling_* transformers/src/transformers/
cp src/mse_optimization.py transformers/src/transformers/
cd transformers
pip install .


