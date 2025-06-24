pip install -e .

python -m pip install torch==2.4.1 
python -m pip install torchvision
python -m pip install qwen-vl-utils
python -m pip install -U flash-attn==2.6.3 --no-build-isolation
python -m spacy download en_core_web_md