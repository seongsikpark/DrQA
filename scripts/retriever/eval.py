#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate the accuracy of the DrQA retriever module."""

import regex as re
import logging
import argparse
import json
import time
import os

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils

#
import matplotlib.pyplot as plt
import numpy as np

#
from memory_profiler import profile

#
from memprof import *

# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None

def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answer, doc_id, match):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    global PROCESS_DB, PROCESS_TOK
    text = PROCESS_DB.get_doc_text(doc_id)
    text = utils.normalize(text)
    if match == 'string':
        # Answer is a list of possible strings
        text = PROCESS_TOK.tokenize(text).words(uncased=True)
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    return False


def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, (doc_ids, doc_scores) = answer_doc

    ret_topk=0
    for doc_id in doc_ids:
        if has_answer(answer, doc_id, match):
            ret_topk = 1
            break

    ret_top1=0
    if len(doc_ids)!=0:
        if has_answer(answer, doc_ids[0], match):
            ret_top1=1

    return ret_top1, ret_topk


def get_score_top1(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, (doc_ids, doc_scores) = answer_doc

    if len(doc_ids) == 0:
        return 0
    elif has_answer(answer, doc_ids[0], match):
        return 1

    return 0

def get_score_topk(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, (doc_ids, doc_scores) = answer_doc

    for doc_id in doc_ids:
        if has_answer(answer, doc_id, match):
            return 1
    return 0





# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
# defalut para.

arg_dataset=None
arg_model=None
arg_doc_db=None
arg_n_docs=5

# for profiling
root_drqa='/home/sspark/Projects/DrQA/'


#arg_dataset=os.path.join(root_drqa,'data/datasets/SQuAD-v1.1-dev.txt')
arg_dataset=os.path.join(root_drqa,'data/datasets/WebQuestions-test.txt')
#arg_dataset=os.path.join(root_drqa,'data/datasets/WikiMovies-test.txt')
#arg_dataset=os.path.join(root_drqa,'data/datasets/CuratedTrec-test.txt')




#arg_model=os.path.join(root_drqa,'data/wikipedia/docs-tfidf-ngram=1-hash=16777216-tokenizer=simple.npz')

# for exe. time measure
#arg_model=os.path.join(root_drqa,'/mnt/external/sspark/Projects/DrQA/data/wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple')

# for accuracy meathre
#arg_model=os.path.join(root_drqa,'/mnt/external/sspark/Projects/DrQA/data/wikipedia/docs-tfidf-ngram=1-hash=16777216-tokenizer=simple.npz')
#arg_model=os.path.join(root_drqa,'/mnt/external/sspark/Projects/DrQA/data/wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
#arg_model=os.path.join(root_drqa,'/mnt/external/sspark/Projects/DrQA/data/wikipedia/docs-tfidf-ngram=3-hash=16777216-tokenizer=simple.npz')
arg_model=os.path.join(root_drqa,'/mnt/external/sspark/Projects/DrQA/data/wikipedia/docs-tfidf-ngram=3-hash=67108864-tokenizer=simple.npz')


#arg_model='/mnt/external/sspark/Projects/DrQA/data/wikipedia/docs-tfidf-ngram=1-hash=1000-tokenizer=simple'
#arg_doc_db=os.path.join(root_drqa,'data/wikipedia/docs.db')
arg_doc_db='/mnt/external/sspark/Projects/DrQA/data/wikipedia/docs.db'
arg_n_docs=5
#arg_num_workers=32
arg_num_workers=1

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                        '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=arg_dataset)
parser.add_argument('--model', type=str, default=arg_model)
parser.add_argument('--doc-db', type=str, default=arg_doc_db,
                    help='Path to Document DB')
parser.add_argument('--tokenizer', type=str, default='regexp')
parser.add_argument('--n-docs', type=int, default=arg_n_docs)
parser.add_argument('--num-workers', type=int, default=arg_num_workers)
parser.add_argument('--match', type=str, default='string',
                    choices=['regex', 'string'])
args = parser.parse_args()

#@profile
def main():

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []
    for line in open(args.dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']
        questions.append(question)
        answers.append(answer)

    # sspark
    num_example=len(questions)
    #num_example=10
    _questions = questions[0:num_example]
    answers = answers[0:num_example]

    print(num_example)

    #print(_questions)
    #print(len(_questions))

    # sl
    questions = []

    # context test
    #for i in range(len(_questions)):
    #    idx_s = max(0, i-2)
    #    idx_e = idx_s+5
    #    tmp_q = ' '.join(_questions[idx_s:idx_e])
    #    questions.append(tmp_q)
    #

    # all token
    questions.append(' '.join(_questions[:]))

    # normal
    #questions = _questions

    #print(questions)
    #questions = []
    #questions.append('NFL')

    #print('questions')
    #print(questions)

    #print(questions)
    #print(answers)

    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    t_load_mat_s=time.time()
    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)
    t_load_mat_e=time.time()


    #logger.info('ssaprk')
    #print(np.max(ranker.doc_freqs))
    #print(np.argmax(ranker.doc_freqs))
    #plt.hist(ranker.doc_freqs, log=True)
    #plt.show()


    logger.info('Ranking...')
    #t_rank_s=time.time()
    closest_docs = ranker.batch_closest_docs(questions, k=args.n_docs, num_workers=args.num_workers)
    answers_docs = zip(answers, closest_docs)



    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )


    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
    get_score_partial = partial(get_score, match=args.match)
    scores = processes.map(get_score_partial, answers_docs)

    scores = np.asarray(scores)
    scores_top1=np.sum(scores[:,0])
    scores_topk=np.sum(scores[:,1])
    tn=scores.shape[0]

    #t_rank_e=time.time()

    filename = os.path.basename(args.dataset)
    stats = (
        "\n" + "-" * 50 + "\n" +
        "{filename}\n" +
        "Examples:\t\t\t{total}\n" +
        "Match % in top 1:\t\t{p1:2.2f}\n" +
        "Match % in top {k}:\t\t{pk:2.2f}\n" +
        "Time Total:\t\t\t{tt:2.4f} (s)\n" +
        "Time Load:\t\t\t{tl:2.4f} (s)\n" +
        "  meta:\t\t\t\t{tlm:2.4f} (s)\n" +
        "  data:\t\t\t\t{tld:2.4f} (s)\n"

    ).format(
        filename=filename,
        total=tn,
        k=args.n_docs,
        p1=(scores_top1 / tn * 100),
        pk=(scores_topk / tn * 100),
        tt=time.time() - start,
        tl=utils.time_load_meta+utils.time_load_data,
        tlm=utils.time_load_meta,
        tld=utils.time_load_data
    )

    print(stats)


#
if __name__ == '__main__':
    main()


