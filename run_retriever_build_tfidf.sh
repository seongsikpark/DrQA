#!/bin/bash


source ./venv/bin/activate


path_db='./data/wikipedia/docs_SQuAD1.db'
path_output='./data/wikipedia'


python ./scripts/retriever/build_tfidf.py $path_db $path_output

