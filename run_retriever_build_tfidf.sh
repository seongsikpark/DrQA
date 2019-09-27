#!/bin/bash


source ./venv/bin/activate


path_db='/mnt/external/sspark/Projects/DrQA/data/wikipedia/docs.db'
#path_db='./data/wikipedia/docs_SQuAD1.db'
path_output='/mnt/external/sspark/Projects/DrQA/data/wikipedia'


#python ./scripts/retriever/build_tfidf.py $path_db $path_output --ngram 1 --hash-size 67108864
python ./scripts/retriever/build_tfidf.py $path_db $path_output --ngram 1 --hash-size 1000
#python ./scripts/retriever/build_tfidf.py $path_db $path_output --ngram 1 --mat-format 'csc'

