"""
This module doesn't do anything yet

"""

import sys
import logging
from flask import Flask, Response, jsonify, request


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s",
    datefmt="%F %H:%M:%S",
    stream=sys.stdout,
)

app = Flask(__name__)


class IrisModel:
    def __init__(self, model_id):
        self.catchphrase = 'you win some you lose some'

    def model(self, input_list):
        prediction_tuple = (
            ['virginica', 0.7],
            ['setosa', 0.2],
            ['versicolor', 0.1]
        )
        return prediction_tuple

    def predict(self, input_list, *args, **kwargs):
        prediction_tuple = self.model(input_list)
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


@app.route(f"/api/predict", methods=["POST"])
def predict():
    app.logger.info(f"Prediction endpoint called")

    input_list = [
        request.args.get('sepal_length'),
        request.args.get('sepal_width'),
        request.args.get('petal_length'),
        request.args.get('petal_width')
    ]

    app.logger.info(f"input list: {input_list}")

    predictions = MODEL.predict(
        input_list
    )

    app.logger.info(f"predictions: {predictions}")

    response = jsonify({"predictions": predictions})
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, threaded=True)
