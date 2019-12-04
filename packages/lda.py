"""
This module contains an implementation of linear discriminant analysis
'from scratch' (i.e. using numpy and pandas but not sklearn).

"""

import numpy as np
import pandas as pd
iris_df = pd.read_csv('data/iris.csv')
X = iris_df[[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
]].copy()
y = iris_df[['species']].copy()





# Compute within class scatter matrices

test = iris_df.groupby('species').mean()

within_class_scatter_matrix = np.zeros((X.shape(1), X.shape[1]))
for c, rows in df.groupby('class'):
    rows = rows.drop(['class'], axis=1)

    s = np.zeros((13,13))
for index, row in rows.iterrows():
        x, mc = row.values.reshape(13,1), class_feature_means[c].values.reshape(13,1)

        s += (x - mc).dot((x - mc).T)

    within_class_scatter_matrix += s

# Compute between class scatter matrices
# Compute eigenvectors and eigenvalues
# Sort the eigenvalues and select the top k
# Obtain the new features by taking dot products of the data and the matrix