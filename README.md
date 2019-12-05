# iris-service

This repo contains a REST micro-service that predicts the species of an iris given 4 observed lengths.

It also contains the exploratory analysis and plotting tools used during the development or the service (these would usually not be checked in, but they are included here for illustration).

# Quickstart

Clone the repo, open a terminal window and navigate to the root folder.
This project uses [Pipenv](https://github.com/pypa/pipenv) to manage python dependencies.

If you do not have pipenv installed, run 

```$ pip install pipenv``` 

to install it

If you do not have pip installed, i think you're probably lost.

Run 

```$ pipenv install``` 

in the root directory of the repo to install all dependencies

Run 

```$ python service.py``` 

to run the flask app containing the service locally. It will run on localhost and listen on port 8080 by default.

Alternatively, run

```$ export FLASK_APP=service.py```


```$ flask run -h 127.0.0.01 -p 1234``` 

to run on an alternative IP and port.

Head to [http://127.0.0.1:8080/api/predict] in your browser or your favorite API client to start making predictions.
