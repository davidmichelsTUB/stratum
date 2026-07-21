from __future__ import annotations
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as _SKCountVectorizer
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import csr_matrix
import logging
from .._config import get_config
from .. import _rust_backend as rb

# File-internal config flags
_DEBUG_INFO = False
logger = logging.getLogger(__name__)
MIN_BLOCK_LEN = 10_000


class RustyCountVectorizer(_SKCountVectorizer):
    """Drop-in CountVectorizer that prefers the Rust fastpath where supported."""

    def __init__(self, n_jobs=None, **kwargs):
        super().__init__(**kwargs)
        cores = os.cpu_count()
        if n_jobs is None:
            self.n_jobs = cores
        elif n_jobs > cores:
            logger.warning(
                f"n_jobs {n_jobs} > core count {cores}, setting n_jobs to {cores}"
            )
            self.n_jobs = cores
        else:
            self.n_jobs = n_jobs

    def _n_chunks(self, corpus):
        blocks = max(1, len(corpus) // MIN_BLOCK_LEN)
        return min(blocks, self.n_jobs)

    def _stopwords_set(self):
        # just use builtin method to cast to list, otherwise
        sw = self.get_stop_words()
        return set() if sw is None else set(sw)

    def _rust_ready(self, fn_name):
        rc = get_config()
        return (
            rc.get("allow_patch", False)
            and rc.get("rust_backend", False)
            and rb.HAVE_RUST
            and getattr(rb, fn_name, None) is not None
        )

    def fit(self, raw_documents, y=None):
        if not self._rust_ready("count_vectorize_fit"):
            logger.warning("Rust disabled, fallback to scikit for fit")
            return super().fit(raw_documents, y)

        corpus = list(raw_documents)
        try:
            vocab = rb.count_vectorize_fit(
                corpus,
                self._stopwords_set(),
                self.token_pattern,
                self._n_chunks(corpus),
            )
            if len(vocab) == 0:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only contain stop words"
                )
        except Exception as e:
            logger.warning(f"Rust count_vectorize_fit failed, falling back: {e}")
            return super().fit(raw_documents, y)

        self.vocabulary_ = vocab
        return self

    def transform(self, raw_documents):
        if not self._rust_ready("count_vectorize_transform"):
            logger.debug("Rust disabled, fallback to scikit for transform")
            return super().transform(raw_documents)

        check_is_fitted(self)

        corpus = list(raw_documents)
        t0 = rb.start_timing()
        try:
            data, indices, indptr = rb.count_vectorize_transform(
                corpus,
                self.vocabulary_,
                self._stopwords_set(),
                self.token_pattern,
                self._n_chunks(corpus),
            )
        except Exception as e:
            logger.warning(f"Rust count_vectorize_transform failed, falling back: {e}")
            return super().transform(raw_documents)
        rb.print_timing("count_vectorize_transform", t0)

        return csr_matrix(
            (data, indices, indptr),
            shape=(len(corpus), len(self.vocabulary_)),
        )

    def fit_transform(self, raw_documents, y=None):
        if not self._rust_ready("count_vectorize_fit_transform"):
            logger.debug("Rust disabled, fallback to scikit for fit_transform")
            return super().fit_transform(raw_documents, y)

        corpus = list(raw_documents)
        t0 = rb.start_timing()
        try:
            vocab, data, indices, indptr = rb.count_vectorize_fit_transform(
                corpus,
                self._stopwords_set(),
                self.token_pattern,
                self._n_chunks(corpus),
            )
            if len(vocab) == 0:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only contain stop words"
                )
        except Exception as e:
            logger.warning(
                f"Rust count_vectorize_fit_transform failed, falling back: {e}"
            )
            return super().fit_transform(raw_documents, y)
        rb.print_timing("count_vectorize_fit_transform", t0)

        self.vocabulary_ = vocab
        return csr_matrix(
            (data, indices, indptr),
            shape=(len(corpus), len(self.vocabulary_)),
        )
