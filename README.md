# iris-service

This repo contains a REST micro-service that predicts the species of an iris given 4 observed lengths.

It also contains the exploratory analysis and plotting tools used during the development or the service (these would usually not be checked in, but they are included here for illustration).

# Quickstart

Clone the repo, open a terminal window and navigate to the root folder.
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

I took some liberties with the output specification because it wasn't fully specified for the case where the model outputs a score for all 3 classes.

The service returns a JSON object with 2 keys: prediction and predictions. The value of prediction is a dictionary with the top predicted species and the estimated probability that the observation belongs to that species. The value of predictions is a list containing 3 dictionaries - one for each species.

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

None of this actually happens, because I changed a bunch of key variables to dummy variables. I didn't fancy actually deploying this on my company's clusters, and I don't have a personal GCP account. As a result, the docker image, circle integration and kubernetes config have not been tested.

# Shortcuts

This code is certainly production capable, however some shortcuts have been taken in the interests of time (logistic regression from scratch isn't quick!) that I would not normally make with production services.

