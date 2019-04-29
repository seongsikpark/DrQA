#!/bin/bash


source ./venv/bin/activate


path_data='./data/datasets/SQuAD-v1.1-dev.txt'


python ./scripts/retriever/eval.py $path_data \
        --num-workers 512 \
        --match 'string'

