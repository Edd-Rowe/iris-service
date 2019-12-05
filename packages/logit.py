"""
This module contains an implementation of multi-class logistic regression
'from scratch' (i.e. using numpy and pandas but not sklearn).

Based on 'Logistic Regression with a Neural Network mindset' in Andrew
Ng's deep learning specialisation, adapted for multi-class classification

Uses neural network principles to set up the logistic regression problem
as a single layer shallow network, then solves it using gradient descent

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_train_test_sets():
    """
    returns the iris dataset, split into predictors, labels, train and test
    sets as 2D numpy arrays

    """
    iris_df = pd.read_csv('data/iris.csv')

    X_df = iris_df[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
    ]].copy()

    # To enable multiple class classification, we need to one-hot encode the
    # species column such that our target vector is of the form [[0, 0, 1]]
    species_df = iris_df[['species']].copy()
    y_df = pd.get_dummies(species_df)

    # Split data into train and test sets - 70% train, 30% test
    # No dev set since we have no hyperparamters to tune
    mask = np.random.rand(len(iris_df)) < 0.7
    # Also convert from dataframes to numpy arrays, and reshape such that
    # a single 'observation' (1 flower) is a column vector.
    train_x = X_df[mask].T.values
    train_y = y_df[mask].T.values
    test_x = X_df[~mask].T.values
    test_y = y_df[~mask].T.values
    return train_x, train_y, test_x, test_y


def softmax(z):
    """
    Compute the softmax vector of each column of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- softmax(z)
    """
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def initialize_with_zeros(dim_0, dim_1):
    """
    Initialise a vector of zeros of shape (dim, 1) for w and b to 0.

    Argument:
    dim -- size of the w vector we want

    Returns:
    w -- initialized vector of shape (dim_0, dim_1)
    b -- initialized scalar (0)
    """
    w = np.zeros([dim_0, dim_1])
    b = 0
    return w, b


def propagate(w, b, X, Y):
    """
    Perform forward propagation and backward propagation once to obtain the
    gradients of the weights and bias with respect to the cost function
    (categorical cross entropy)

    Arguments:
    w -- weights, a numpy array of size (num_features, num_classes)
    b -- bias, a scalar
    X -- data of size (num_features, num_samples)
    Y -- true "label" vector
         ([1,0,0], mapping to [setosa, versicolot, virginica])

    Return:
    cost -- categorical cross entropy cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    """
    m = X.shape[1]
    # Forward propagation
    A = softmax(np.dot(w.T, X) + b)

    # The cost function for logistic regression - categorical cross entropy
    cost = -(1/m) * (Y*np.log(A) + (1-Y)*np.log(1-A)).sum()

    # Backward propagation
    dw = 1/m * np.dot(X, (A - Y).T)
    db = 1/m * (A-Y).sum()

    grads = {"dw": dw, "db": db}

    return grads, cost


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression
    parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_features, num_classes)
    b -- bias, a scalar
    X -- data of size (num_features, num_samples)

    Returns:
    Y_prediction -- a numpy array containing all predictions
    ([1,0,0], mapping to [setosa, versicolot, virginica])
    for the examples in X

    """
    # Iniitalise Y_prediction vetor
    m = X.shape[1]
    Y_prediction = np.zeros((3, m))
    w = w.reshape(X.shape[0], 3)
    A = softmax(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        # Simply select the highest probability
        if A[0, i] == A[:, i].max():
            Y_prediction[:, i] = [1, 0, 0]
        elif A[1, i] == A[:, i].max():
            Y_prediction[:, i] = [0, 1, 0]
        else:
            Y_prediction[:, i] = [0, 0, 1]

    assert(Y_prediction.shape == (3, m))

    return Y_prediction


def optimize(w, b, X, Y, num_iterations, learning_rate, test_x, test_y):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_features, num_classes)
    b -- bias, a scalar
    X -- data of shape (num_features, num_samples)
    Y -- true "label" vector
         ([1,0,0], mapping to [setosa, versicolot, virginica])
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    test_x, test_y are the testing set features and labels, used to log
    the test set accuracy every 100 iterations

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias
    with respect to the cost function
    train_accs, test_accs -- list of all prediction accuracies of the model
    logged during optimisation (once per hundred iterations)

    """
    train_accs = []
    test_accs = []
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule weights and bias
        w = w - learning_rate*dw
        b = b - learning_rate*db
        # Record the costs
        if i % 100 == 0:
            Y_prediction_test = predict(w, b, test_x)
            Y_prediction_train = predict(w, b, X)
            train_accs.append(
                100 - np.mean(np.abs(Y_prediction_train - Y)) * 100
            )
            test_accs.append(
                100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100
            )

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, train_accs, test_accs


def model(train_x, train_y, test_x, test_y, num_iterations=2000,
          learning_rate=0.5):
    """
    Builds the logistic regression model using gradient descent

    Arguments:
    train_x -- training set represented by a numpy array of shape
    (num_features, num_samples)

    train_y -- training labels represented by a numpy array (vector) of
    shape (num_classes, num_samples)

    test_x -- test set represented by a numpy array of shape
    (num_features, num_samples)

    test_y -- test labels represented by a numpy array (vector) of
    shape (num_classes, num_samples)

    num_iterations -- hyperparameter representing the number of iterations
    to optimize the parameters

    learning_rate -- hyperparameter representing the learning rate used
    in the update rule of optimize()


    Returns:
    MODEL -- dictionary containing information about the model.
    """
    # initialize parameters with zeros
    w, b = initialize_with_zeros(train_x.shape[0], train_y.shape[0])

    # Optimise model using gradient descent
    parameters, grads, train_accs, test_accs = optimize(
        w, b, train_x, train_y, num_iterations, learning_rate, test_x, test_y
    )

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, test_x)
    Y_prediction_train = predict(w, b, train_x)

    MODEL = {
            "Y_prediction_test": Y_prediction_test,
            "Y_prediction_train": Y_prediction_train,
            "w": w,
            "b": b,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations,
            "train_accs": train_accs,
            "test_accs": test_accs
    }
    return MODEL


def plot_learning_curve(MODEL):
    """
    Plots the learning curve - the train set and test set error vs number of
    iterations that was saved during MODEL training.

    As a shortcut, this was used in development to quickly find some decent
    values for the learning rate and number of iterations.

    """
    train_accs = np.squeeze(MODEL['train_accs'])
    test_accs = np.squeeze(MODEL['test_accs'])
    fig, ax = plt.subplots()
    ax.plot(100 - train_accs, label='Training set error')
    ax.plot(100 - test_accs, label='Test set error')
    ax.set_ylabel('Prediction error (%)')
    plt.legend()
    ax.set_xlabel('Hundreds of iterations')
    ax.set_title("Learning rate =" + str(MODEL["learning_rate"]))


def load_iris_and_return_model(num_iterations, learning_rate):
    """
    returns MODEL - a dictonary containing all the necessary information
    about the trained logistic regression model

    """
    train_x, train_y, test_x, test_y = get_train_test_sets()
    MODEL = model(
        train_x, train_y, test_x, test_y, num_iterations=num_iterations,
        learning_rate=learning_rate
    )
    return MODEL


if __name__ == '__main__':
    # Quick & hacky hyperparameter tuning
    num_iterations = 50000
    learning_rate = 0.001
    MODEL = load_iris_and_return_model(
        num_iterations=50000, learning_rate=0.001
    )
    plot_learning_curve(MODEL)
