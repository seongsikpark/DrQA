#!/bin/bash


source ./venv/bin/activate

path_data='./data/datasets/SQuAD-v1.1-dev.json'
prediction='./tmp/SQuAD-v1.1-dev-default.preds'


python ./scripts/reader/official_eval.py $path_data $prediction

