#!/bin/bash


source ./venv/bin/activate

path_data='./data/datasets/SQuAD-v1.1-dev.json'


python ./scripts/reader/predict.py $path_data \
        --gpu 0 \
        --official \
        --tokenizer corenlp
