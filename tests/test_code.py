# -*- coding: utf-8 -*-

import numpy as np
from packages import logit


def test_softmax_sums_to_one():
    # Softmax of a random array sums to one (within floating point error)
    softmax_sum = logit.softmax(np.array([
        [np.random.rand()],
        [np.random.rand()],
        [np.random.rand()]
    ])).sum()
    expected_result = 1
    difference = expected_result - softmax_sum
    assert abs(difference) < 0.000001


def test_softmax_gives_expected_value():
    # Softmax returns expected value for a hard-coded test case
    softmax = logit.softmax(np.array([
        [3],
        [1],
        [2]
    ]))
    expected_result = np.array([
        [0.665241],
        [0.0900306],
        [0.244728]
    ])
    difference = softmax-expected_result
    assert abs(difference.sum()) < 0.000001


def test_iris_dataset_loaded_completely():
    # get_train_test_sets returns arrays of the expected shape
    train_x, train_y, test_x, test_y = logit.get_train_test_sets()
    assert train_x.shape[0] == 4
    assert test_x.shape[0] == 4
    assert train_y.shape[0] == 3
    assert test_y.shape[0] == 3
    assert train_x.shape[1] + test_x.shape[1] == 150
    assert train_y.shape[1] + test_y.shape[1] == 150


def test_model_returns_expected_dict():
    # load_iris_and_return_model returns the expected dictionary
    # (it also executes without throwing errors)
    MODEL = logit.load_iris_and_return_model(30, 0.001)
    assert len(MODEL) == 8
