#!/bin/bash


source ./venv/bin/activate



path_data='./data/datasets/SQuAD-v1.1-dev.txt'
path_doc_db='./data/wikipedia/docs_SQuAD1.db'
path_retriver_model='./data/wikipedia/docs_SQuAD1-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'


python ./scripts/pipeline/predict.py $path_data \
        --doc-db $path_doc_db \
        --retriever-model $path_retriver_model \
        --gpu 0 \
        --tokenizer corenlp
