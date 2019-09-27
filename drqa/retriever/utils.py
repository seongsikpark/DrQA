#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Various retriever utilities."""

import regex
import unicodedata
import numpy as np
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32

from collections import OrderedDict

import os
import time

# ------------------------------------------------------------------------------
# Sparse matrix saving/loading helpers.
# ------------------------------------------------------------------------------
time_load_meta=0.0
time_load_data=0.0


def save_sparse_csr(filename, matrix, metadata=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)

def save_sparse_csc(filename, matrix, metadata=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)



def load_sparse_csr(filename):
    loader = np.load(filename)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None

# .npz load
def load_sparse(filename):
    global time_load_meta
    global time_load_data

    time_meta_s=time.time()
    loader = np.load(filename)

    #meta = loader['metadata'].item(0) if 'metadata' in loader else None
    meta = loader['metadata'].item(0)
    #mat_format = meta['mat_format'] if 'mat_format' in meta else None

    time_meta_e=time.time()

    time_data_s=time.time()
    #if mat_format =='csc':
    #    matrix = sp.csc_matrix((loader['data'], loader['indices'],loader['indptr']), shape=loader['shape'])
    #else:
    #    matrix = sp.csr_matrix((loader['data'], loader['indices'],loader['indptr']), shape=loader['shape'])
    matrix = sp.csr_matrix((loader['data'], loader['indices'],loader['indptr']), shape=loader['shape'])
    time_data_e=time.time()

    #print('time meta: %f'%(time_meta_e-time_meta_s))
    #print('time data: %f'%(time_data_e-time_data_s))

    time_load_meta = time_meta_e-time_meta_s
    time_load_data = time_data_e-time_data_s

    return matrix, meta


def load_sparse_meta(filename):
    global time_load_meta

    #loader = np.load(filename, mmap_mode='r')
    #meta = loader['metadata'].item(0)
    time_s=time.time()
    meta=np.load(os.path.join(filename,'metadata.npy')).item(0)
    time_e=time.time()

    #print('time meta: %f'%(time_e-time_s))
    time_load_meta += time_e-time_s
    return meta

def load_sparse_idx(filename, csr):
    global time_load_data

    time_load_data_s=time.time()
    #print('np.load start')
    #loader = np.load(filename)
    #loader = np.load(filename, mmap_mode='r')

    loader = OrderedDict()
    loader['data']=np.load(os.path.join(filename,'data.npy'),mmap_mode='r')
    loader['indices']=np.load(os.path.join(filename,'indices.npy'),mmap_mode='r')
    loader['indptr']=np.load(os.path.join(filename,'indptr.npy'),mmap_mode='r')
    loader['shape']=np.load(os.path.join(filename,'shape.npy'),mmap_mode='r')


    #loader['data']=np.load(os.path.join(filename,'data.npy'))
    #loader['indices']=np.load(os.path.join(filename,'indices.npy'))
    #loader['indptr']=np.load(os.path.join(filename,'indptr.npy'))
    #loader['shape']=np.load(os.path.join(filename,'shape.npy'))

    #print('np.load done')

    shape=loader['shape']
    #print(shape)

    indptr = np.zeros(shape[0]+1,dtype=np.int32)

    #print(indptr.shape)

    data=[]
    indices=[]

    #csr.indices=[9160841]

    #csr.indices = np.random.randint(0,1677215,size=100)

    #sind=np.random.randint(0,1677115,size=1)
    #csr.indices = np.arange(sind,sind+100)

    # ori
    for idx in csr.indices:
        #print(idx)
        idx_start = loader['indptr'][idx]
        num_load = loader['indptr'][idx+1]-idx_start
        idx_end = idx_start+num_load
        #print(num_load)
        #data.append(loader['data'][idx_start:idx_end])
        #indices.append(loader['indices'][idx_start:idx_end])

        data.extend(loader['data'][idx_start:idx_end])
        indices.extend(loader['indices'][idx_start:idx_end])

        indptr[idx+1]=num_load
        #last_ind_ptr
        #indptr[last_ind_ptr+1:idx+1]

    # rand
    #csr.indices = np.random.randint(0,1677215,size=500)
    #for idx in csr.indices:
    #    data.extend(loader['data'][idx:idx+1])
    #    indices.extend(loader['indices'][idx:idx+1])


    # seq
    #idx = np.random.randint(0,1677215,size=1)[0]
    #data.extend(loader['data'][idx:idx+500])
    #indices.extend(loader['indices'][idx:idx+500])


    indptr=np.cumsum(indptr)
    #print(indptr)
    #print(data)
    #print(indices)
    #print(shape)
    #print(indptr)


    #data=np.array(data,dtype=np.float32)
    #indices=np.array(indices,dtype=np.int32)

    #matrix = sp.csr_matrix((loader['data'], loader['indices'],loader['indptr']), shape=loader['shape'])
    #matrix = sp.csr_matrix(data, indices, indptr, shape, dtype=float)

    #loader_ori=np.load('/home/sspark/Projects/DrQA/data/wikipedia/docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
    #loader_ori=np.load(filename+'.npz')
    #indices_ori=loader_ori['indices']

    #print(len(indices_ori))
    #print(len(indices))


    #data=np.asarray(data)
    #indices=np.asarray(indices)
    #indptr=np.asarray(indptr)

#    print('here')
#    print(csr.indices)
#    ind = csr.indices[0]
#    print(loader_ori['indptr'][ind])
#    ind_s=loader_ori['indptr'][ind]
#    ind_e=loader_ori['indptr'][ind+1]
#    ind_n=ind_e-ind_s
#    print(loader_ori['data'][ind_s:ind_s+ind_n][0:10])
#    print(data[0:10])
#    print(type(loader_ori['indices'][ind_s:ind_s+ind_n][0:10]))
#    print(type(indices[0:10]))
#
#    print(indptr)

    #matrix = sp.csr_matrix((loader_ori['data'], loader_ori['indices'],loader_ori['indptr']), shape=loader_ori['shape'])
    matrix = sp.csr_matrix((data, indices, indptr), shape=shape)


    time_load_data_e=time.time()

    time_load_data += time_load_data_e-time_load_data_s
    #print(time_load_data)
    return matrix





# ------------------------------------------------------------------------------
# Token hashing.
# ------------------------------------------------------------------------------


def hash(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    ret = murmurhash3_32(token, positive=True)

    if ret >= num_buckets:
        ret=ret%num_buckets
        #print('duplicate')

    #if ret == 12088341:
    #    print('token')
    #    print(token)
    #    #assert False

    return ret


# ------------------------------------------------------------------------------
# Text cleaning.
# ------------------------------------------------------------------------------


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren',
    'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"
}


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def filter_word(text):
    """Take out english stopwords, punctuation, and compound endings."""
    text = normalize(text)
    if regex.match(r'^\p{P}+$', text):
        return True
    if text.lower() in STOPWORDS:
        return True
    return False


def filter_ngram(gram, mode='any'):
    """Decide whether to keep or discard an n-gram.

    Args:
        gram: list of tokens (length N)
        mode: Option to throw out ngram if
          'any': any single token passes filter_word
          'all': all tokens pass filter_word
          'ends': book-ended by filterable tokens
    """
    filtered = [filter_word(w) for w in gram]
    if mode == 'any':
        return any(filtered)
    elif mode == 'all':
        return all(filtered)
    elif mode == 'ends':
        return filtered[0] or filtered[-1]
    else:
        raise ValueError('Invalid mode: %s' % mode)

def get_field(d, field_list):
    """get the subfield associated to a list of elastic fields 
        E.g. ['file', 'filename'] to d['file']['filename']
    """
    if isinstance(field_list, str):
        return d[field_list]
    else:
        idx = d.copy()
        for field in field_list:
            idx = idx[field]
        return idx
