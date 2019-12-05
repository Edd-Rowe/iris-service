"""
This module defines the IrisModel class, which trains the logistic regression
model in packages.logit using gradient descent and can subsequntly serve
predictions to an endpoint via a Flask.

This module also contains the flask configuration and defines the endpoints
and their response.

Running this module directly with $ python service.py will instantiate an
instance of IrisModel, and run the service locally on localhost:8080

Running this model with
$ export FLASK_APP=service.py
$ flask run -h ***.*.*.* -p ****
runs the service using an IP and port of your choice

"""

import sys
import logging
import numpy as np
from flask import Flask, Response, jsonify, request
from packages.logit import load_iris_and_return_model, softmax


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%F %H:%M:%S",
    stream=sys.stdout,
)

app = Flask(__name__)

class IrisModel:
    def __init__(self, model_id):
        # In the interest of time, we cheat a little bit and train the
        # model when the IrisModel class is instantiated
        # We have also done quick offline hyperparamter tuning for alpha and n
        self.model_id = model_id
        self.model = load_iris_and_return_model(
            num_iterations=50000, learning_rate=0.001
        )

    def predict(self, X, *args, **kwargs):
        w = self.model.get('w')
        b = self.model.get('b')
        w = w.reshape(X.shape[0], 3)
        A = softmax(np.dot(w.T, X) + b)
        prediction_tuple = (
            ['setosa', A[0, 0]],
            ['versicolor', A[1, 0]],
            ['virginica', A[2, 0]]
        )
        return [
            {
                "class": p[0],
                "score": p[1]
            }
            for p in prediction_tuple
        ]


MODEL_VERSION = '0.0.1'
MODEL = IrisModel(MODEL_VERSION)


@app.route(f"/healthz", methods=["GET"])
def health():
    app.logger.info(f"healthz endpoint called")
    return Response(status=200)


@app.route(f"/api/predict", methods=["GET"])
def predict():
    app.logger.info(f"Prediction endpoint called")
    try:
        input_array = np.array([[
            float(request.args.get('sepal_length')),
            float(request.args.get('sepal_width')),
            float(request.args.get('petal_length')),
            float(request.args.get('petal_width'))
        ]]).T
    except TypeError:
        return """Invalid input"""
    app.logger.info(f"input array: {input_array}")

    predictions = MODEL.predict(
        input_array
    )
    # Find class with the greatest score
    scores = [
        predictions[0].get('score'),
        predictions[1].get('score'),
        predictions[2].get('score')
    ]
    max_index = scores.index(max(scores))
    prediction = predictions[max_index]

    app.logger.info(f"prediction: {prediction}, predictions: {predictions}")

    response = jsonify({"prediction": prediction, "predictions": predictions})
    return response


def run_app():
    app.run(host="127.0.0.1", port=8080, threaded=True)


if __name__ == "__main__":
    run_app()
