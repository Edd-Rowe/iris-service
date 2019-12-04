"""
This module contains an implementation of multi-class logistic regression
'from scratch' (i.e. using numpy and pandas but not sklearn).

"""

import numpy as np
import pandas as pd
iris_df = pd.read_csv('data/iris.csv')
X = iris_df[[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
]].copy()
y = iris_df[['species']].copy()
