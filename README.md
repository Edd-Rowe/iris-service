# iris-service

This repo contains a RESTful micro-service that predicts the species of an iris given the lengths and widths of its petals and sepals.

Prediction is carried out using a logistic regression model implemented 'from scratch' (not using sklearn).

It also contains the exploratory analysis and plotting tools used during the development or the service (these would usually not be checked in, but they are included here for illustration).

This repo is intended to showcase competency in a bunch of data science skills: 
- Python
- Machine learning
- RESTful microservices
- Deployment
- Testing
- Git(hub) & repository structure

# Quickstart

Clone the repo, open a terminal window and navigate to the root directory.

This project uses [Pipenv](https://github.com/pypa/pipenv) to manage python dependencies.

Install pipenv if you haven't already:

```$ pip install pipenv``` 

Install all project dependencies:

```$ pipenv install``` 

Run the flask app containing the service locally on localhost and the default port:

```$ python service.py``` 

Alternatively, pick a different IP and port:

```$ export FLASK_APP=service.py```


```$ flask run -h 127.0.0.01 -p 1234``` 

Head to http://127.0.0.1:8080/api/predict?sepal_length=5&sepal_width=0.3&petal_length=5&petal_width=3.5 in your browser or your favorite API client to see an example prediction.

Edit the query string in your browser or API client to predict the species of your own personal iris.

# Output

The service returns a JSON object with 2 keys: `prediction` and `predictions`. `prediction` contains the most likely class and the confidence score for the class (which can be interpreted as the probability that your iris if of this species). `predictions` contains all 3 classes with their confidence scores, which sum to one.

```json
{
    "prediction": {
        "class": "virginica",
        "score": 0.9999794402832914
    },
    "predictions": [
        {
            "class": "setosa",
            "score": 5.92233161988833e-10
        },
        {
            "class": "versicolor",
            "score": 0.000020559124475387382
        },
        {
            "class": "virginica",
            "score": 0.9999794402832914
        }
    ]
}
```

# Deployment

This repo contains configuration files for Docker & circleci and a manifest for kubernetes.
If we hooked circleci into this repo, whenever a commit was pushed to master branch, the build-and-deploy workflow defined in .circleci/config.yml would trigger.

This workflow builds a docker image with python 3.7 & installs the dependencies in pipenv. It stores the docker image in a Google Container Registry, which is then deployed on a kubernetes cluster.

None of this actually happens, because I changed a bunch of key variables to dummy variables. As a result, the docker image, circle integration and kubernetes config have not been tested.

# Model

`explore_data.py` contains some explanatory analysis of the dataset, including this scatter plot matrix of the 4 features with color coded species labels. As documented in that script, the linear decision boundaries and not-quite-normal distributions of the features suggest logistic regression is a good model to fit the data.

![scatter plot matrix of iris dataset](https://github.com/Edd-Rowe/iris-service/blob/master/images/scatter_matrix.png)

`packages/logit.py` implements logistic regression 'from scratch' - i.e. using numpy and pandas but not using sklearn.

The logistic regression cost function (categorical cross entropy) is minimised using gradient descent on a matrix of weights and a scalar bias term.

The learning curve shows that test set error is minimised after ~17,000 iterations using a learning rate of 0.001.

![learning curve of gradient descent](https://github.com/Edd-Rowe/iris-service/blob/master/images/learning_curve.png)

# Tests

There are basic unit tests for `logit.py` and some tests for the service which require you to manually run the service first.

If you do not have pytest installed, you can install it with:

``` $ pipenv install pytest --dev```

Alternatively, install all --dev packages:

``` $ pipenv install -- dev```

Run the service on localhost port 8080 as described in `Quickstart` above. Run the tests by running:

``` $ python -m pytest```

# Shortcuts

This code is certainly production capable, however some shortcuts have been taken since this is just a demonstration.

- The logistic regression model that generates the predictions is trained at runtime (when the IrisModel class is instantiated by service.py). Usually, one would train my model and tune hyperparameters in a seperate module and save the weights to be loaded later.

- I perormed some quick and dirty hyperparameter tuning to find workable values for the learning rate and the number of iterations by manually running my script. I only split my training data into train/test sets, not train/dev/test.

- The service runs on the flask default development server. This is not intended to be used for production, just local testing like described in `Quickstart`. Usually, one would use a production WSGI server like Waitress for anything in production. If the warning annoys you, you can avoid it by running the following before you run the service:

```
$ export FLASK_ENV=development
```

- The `packages` folder has no packages, and in fact only has one module (and itself has no `__init__.py`)

- The unit tests for the code do not cover close to all of the code, they just cover a few illustrative examples.

- The tests for the service do not use Flask's built in testing mode - instead you must run the service somewhere else (in another terminal window or k8s) and then hit the endpoint using `requests`

- There are no unit tests triggered by commits that excecute in circleci. Usually, one would write unit tests that trigger on commit to ANY branch (not just master), that excecute in circleci, in the same workflow as the docker build. 

- In general, there is less error handling and fewer assertions in the code than one would normally write in production software.
