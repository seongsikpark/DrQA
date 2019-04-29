#!/bin/bash


source ./venv/bin/activate

path_data='./data/datasets/SQuAD-v1.1-dev.txt'


python ./scripts/pipeline/predict.py $path_data \
        --gpu 0
        --tokenizer corenlp
