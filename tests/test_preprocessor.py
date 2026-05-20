"""Tests for the MediGuard preprocessing pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.data.simulator    import generate_dataset
from src.data.preprocessor import MediGuardPreprocessor, make_sequences


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_normal=500, n_per_attack=50, seed=0)


def test_dataset_shape(dataset):
    assert len(dataset) > 0
    assert "label" in dataset.columns
    assert "attack_type" in dataset.columns


def test_preprocessor_fit(dataset):
    prep = MediGuardPreprocessor()
    X_tr, X_te, y_tr, y_te = prep.fit_transform(dataset)
    assert X_tr.shape[0] > 0
    assert X_tr.shape[1] == X_te.shape[1]
    assert len(y_tr) == X_tr.shape[0]


def test_preprocessor_transform(dataset):
    prep = MediGuardPreprocessor()
    prep.fit_transform(dataset)
    X = prep.transform(dataset.head(10))
    assert X.shape == (10, len(prep.feature_cols_))


def test_sequences(dataset):
    prep = MediGuardPreprocessor()
    X, _, y, _ = prep.fit_transform(dataset)
    X_seq, y_seq = make_sequences(X, y, seq_len=10)
    assert X_seq.shape == (len(X) - 10, 10, X.shape[1])
    assert len(y_seq) == len(X_seq)


def test_no_nan(dataset):
    prep = MediGuardPreprocessor()
    X_tr, X_te, _, _ = prep.fit_transform(dataset)
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_te).any()
