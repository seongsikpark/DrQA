#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from . import DEFAULTS
from .. import tokenizers

#
import matplotlib.pyplot as plt



logger = logging.getLogger(__name__)


class TfidfDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, tfidf_path=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        tfidf_path = tfidf_path or DEFAULTS['tfidf_path']

        if tfidf_path.endswith('.npz'):
            f_suffix_npz=True

            logger.info('Loading: %s' % tfidf_path)
            #matrix, metadata = utils.load_sparse_csr(tfidf_path)
            matrix, metadata = utils.load_sparse(tfidf_path)
            logger.info('Loading end: %s end' % tfidf_path)

            self.doc_mat = matrix
        else:
            f_suffix_npz=False

            metadata = utils.load_sparse_meta(tfidf_path)

            self.doc_mat = None

        self.tfidf_path = tfidf_path
        self.mat_format = metadata['mat_format'] if 'mat_format' in metadata else None

        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

        #
        self.f_suffix_npz=f_suffix_npz


        #print('test')
        #print(len(matrix.indptr))

        print(self.doc_freqs.shape)
        print(np.max(self.doc_freqs))
        print(np.mean(self.doc_freqs))
        print(np.min(self.doc_freqs))



    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)

        #print('spvec')
        #print(spvec)

        if self.mat_format=='csc':
            #res = spvec.transpose() * self.doc_mat
            #res = self.doc_mat
            #res = self.doc_mat.dot(spvec)
            #res = spvec * self.doc_mat
            #res = spvec.dot(self.doc_mat)
            #res = self.doc_mat.dot(spvec)
            #res = self.doc_mat.multiply(spvec)

            #res = self.doc_mat.multiply(spvec).sum(0)
            #res = self.doc_mat.multiply(spvec)
            #res = res.sum(0)
            #res = spvec * self.doc_mat
            #print(res.shape)

            res = spvec * self.doc_mat
        else:

            if self.f_suffix_npz:
                # original
                doc_mat = self.doc_mat
            else:
                doc_mat = utils.load_sparse_idx(self.tfidf_path,spvec)
            res = spvec * doc_mat


        #print(query)
        #print(spvec)
        #print(spvec.shape)

        #print(self.doc_mat[10])
        #print(self.doc_mat[100])
        #print(self.doc_mat.shape)
        #print(self.doc_mat[0,:])
        #print(self.doc_mat.nnz)

        #print('ngrams: %d'%(self.ngrams))
        #print('hash_size: %d'%(self.hash_size))
        #print('num_docs: %d'%(self.num_docs))
        #print('doc_freqs shape: %s'%(self.doc_freqs.shape))
        #print(self.doc_freqs.shape)
        #print(self.doc_dict[0])
        #print(len(self.doc_freqs))

#        print("non zero")
#        print(np.count_nonzero(self.doc_freqs))
#
#        print("total sum")
#        print(np.sum(self.doc_freqs))
#
#        print("max")
#        print(np.max(self.doc_freqs))
#
#        print("mean")
#        print(np.mean(self.doc_freqs))
#
#        print("min")
#        print(np.min(self.doc_freqs))
#
#        print("std")
#        print(np.std(self.doc_freqs))

        ##
        #print(self.doc_mat.shape)
        #print("max")
        #print(np.max(self.doc_mat))
        #print("mean")
        #print(np.mean(self.doc_mat))
        #print("min")
        #print(np.min(self.doc_mat))
        #print("std")
        #print(np.std(self.doc_mat))
        #print("non_zero")
        #print(np.count_nonzero(self.doc_mat))

        #Ns = self.doc_freqs
        #dfs = np.log((Ns + 0.5)/(self.num_docs - Ns + 0.5))
        #dfs[dfs < 0] = 0

        # TF-IDF*IDF
        #data = np.multiply(self.doc_mat, dfs)
        #print('TF max')
        #print(np.max(data))

        #sum_0=np.sum(self.doc_mat,axis=0)
        #print(np.sum(self.doc_mat,axis=1))

        #print(sum_0.shape)

        #print(self.doc_dict.shape())
        #print(len(self.doc_dict[0]))
        #print(len(self.doc_dict[1]))

        #num_zero = len(self.doc_freqs)-np.count_nonzero(self.doc_freqs)
        #print(num_zero)
        #print(np.count_nonzero(self.doc_freqs)/len(self.doc_freqs))

        #print(self.doc_freqs)






        #for idx in range(self.doc_mat.nnz):
        #    print(self.doc_mat[idx])

        #print(res)

        #print(res.shape)

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]


        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))

        print(len(words))

        #words = self.parse(query)
        wids = [utils.hash(w, self.hash_size) for w in words]

        #print(query)
        #print(len(words))
        #print(words)
        #print(wids)

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)

                if self.mat_format=='csc':
                    ret = sp.csc_matrix((1, self.hash_size))
                else:
                    ret = sp.csr_matrix((1, self.hash_size))
                return ret

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0


        # TF-IDF
        data = np.multiply(tfs, idfs)


        if self.mat_format=='csc':
            # One col, sparse csc matrix
            indptr = np.array([0,len(wids_unique)])
            print(wids_counts)
            print(wids_unique)
            print(indptr)
            #indptr = np.array([0, self.num_docs])
            spvec = sp.csc_matrix((data, wids_unique, indptr), shape=(self.hash_size, 1))
            #spvec = sp.csc_matrix((data, wids_unique, indptr), shape=(1, self.hash_size))
            #spvec = sp.csc_matrix((data, 0, wids_unique), shape=(1, self.hash_size))
            #spvec = sp.csc_matrix(data, (0,wids_unique), shape=(1,self.hash_size))
            #spvec = sp.csc_matrix((data, wids_unique, indptr), shape=(self.num_docs, 1))

            #indptr = np.array([0, len(wids_unique)])
            #spvec = sp.csr_matrix((data, wids_unique, indptr), shape=(1, self.hash_size))
        else:
            # One row, sparse csr matrix
            indptr = np.array([0, len(wids_unique)])
            spvec = sp.csr_matrix((data, wids_unique, indptr), shape=(1, self.hash_size))


        return spvec
