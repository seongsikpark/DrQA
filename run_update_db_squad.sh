#!/bin/bash


source ./venv/bin/activate


path_wiki_db='./data/wikipedia/docs_SQuAD1.db'
path_squad_data='./data/datasets/SQuAD-v1.1-dev.json'


python ./scripts/retriever/update_db_squad.py $path_wiki_db $path_squad_data

