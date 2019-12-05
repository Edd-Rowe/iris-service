# -*- coding: utf-8 -*-

import requests
import numpy as np


def test_api_gives_obvious_prediction():
    """
    This comibination of features is obviously an example of virginica.
    No matter how bad your hyperparamters are, if this does not return
    virginica you've probably added a bug

    """
    # Start the service manually in terminal before testing
    url = 'http://127.0.0.1:8080/api/predict?sepal_length=5&sepal_width=0.3&petal_length=5&petal_width=3.5'  # noqa E501
    req = requests.get(
            url=url,
            timeout=(10, 15)
        )
    result = req.json()
    prediction_class = result.get('prediction').get('class')
    assert prediction_class == 'virginica'


def test_api_scores_sum_to_one():
    # service should return 3 scores that sum to 1 (within fp error)
    url = f"""http://127.0.0.1:8080/api/predict?sepal_length={
        np.random.rand()
    }&sepal_width={np.random.rand()}&petal_length={
        np.random.rand()
    }&petal_width={np.random.rand()}"""
    req = requests.get(
        url=url,
        timeout=(10, 15)
    )
    result = req.json()
    summed_scores = sum(
        result.get('predictions')[i].get('score') for i in range(
            len(result.get('predictions'))
        )
    )
    assert abs(summed_scores - 1) < 0.0000001
