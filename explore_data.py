"""
This script explores the provided IRIS dataset to look for data quality
issues and any obvious patterns that might suggest what models are suitable
for predicting species

This script is not intended to be imported as a module, and would usually
be .gitignored, but i'll leave it in for the purposes of this excercise.

I'll also pretend that I don't know what the Iris dataset is and that it's
more easily downloaded and toyed with via sklearn.

"""
import pandas as pd
import seaborn as sns
iris_df = pd.read_csv('data/iris.csv')

# Check for null values in the iris dataset
print(iris_df.isna().any())  # no null values :)
# Check datatypes of columns - any mixed columns? Numbers that are strings?
print(iris_df.dtypes)  # nope, looks good to me :)

# Check for outliers or abnormalities in the continous variables
iris_df.describe()  # again, looks good. No issues here.

# Visualise the scatterplot matrix of the 4 continuous independent variables
sns.set(style="ticks")
sns.pairplot(iris_df, hue="species")

"""
We note that versicolor and virginica are not in seperable clusters.
This suggests that unsupervised approaches might not be very effective.

There does seem to be some explanatory power in each of sepal_length,
petal_length and petal_width that discriminates versicolor from virginica.
In additional, we have access to species labels for our entire dataset,
so perhaps we should build a supervised ML model.

Inspecting the individual pairplots suggests the decision boundaries are all
linear, so a neural net is overkill. An implementaion of linear discriminant
analysis seems appropriate here.

"""
