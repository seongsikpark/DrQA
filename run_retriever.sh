#!/bin/bash


source ./venv/bin/activate


path_data='./data/datasets/SQuAD-v1.1-dev.txt'
#path_data='./data/datasets/WebQuestions-test.txt'

#path_model='./data/wikipedia/docs-tfidf-ngram=1-hash=16777216-tokenizer=simple.npz'
#path_model='./data/wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
#path_model='./data/wikipedia/docs-tfidf-ngram=3-hash=16777216-tokenizer=simple.npz'
#path_model='./data/wikipedia/docs-tfidf-ngram=1-hash=16777216-tokenizer=simple-csc.npz'
#path_model='./data/wikipedia/docs-tfidf-ngram=1-hash=100-tokenizer=simple-csc.npz'

#path_model='./data/wikipedia/docs-tfidf-ngram=1-hash=67108864-tokenizer=simple.npz'
#path_model='./data/wikipedia/docs-tfidf-ngram=3-hash=67108864-tokenizer=simple.npz'
#path_model='./data/wikipedia/docs_SQuAD1-tfidf-ngram=1-hash=16777216-tokenizer=simple.npz'



path_model='/mnt/external/sspark/Projects/data/wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple'

path_doc_db='./data/wikipedia/docs.db'
#path_doc_db='./data/wikipedia/docs_SQuAD1.db'


docs_topk=5

#num_workers=None

python -u ./scripts/retriever/eval.py
        --dataset $path_data \
        --model $path_model \
        --doc-db $path_doc_db \
        --n-docs $docs_topk \
        --num-workers 1




