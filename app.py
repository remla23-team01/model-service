"""
Module that contains the Flask application and the endpoints to serve the
model.
"""

import re

import joblib
import nltk
import numpy as np
from flasgger import Swagger
from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
from flasgger import Swagger
from remla01_lib import mlSteps
import numpy as np
# from logger.custom_logger import CustomFormatter
# import logging

"""FLASK"""""
app = Flask(__name__)

# CORS
CORS(app)

# SWAGGER
swagger = Swagger(app)

# LOGGING
# create logger with custom formatter
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# # create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# ch.setFormatter(CustomFormatter())
# logger.addHandler(ch)


# Metrics
NUMBER_OF_REQUESTS = 0
NUMBER_OF_POSTIVE_PREDICTIONS = 0
NUMBER_OF_NEGATIVE_PREDICTIONS = 0
NUMBER_OF_CORRECT_PREDICTIONS = 0
NUMBER_OF_INCORRECT_PREDICTIONS = 0

# Reviews
reviews = []


class Review:
    """
    Class that represents a review.
    """

    id: int
    review: str
    predicted: int
    actual: int

    def __init__(self, id: int, review: str, predicted: int, actual: int = -1):
        self.id = id
        self.review = review
        self.predicted = predicted
        self.actual = actual

    def get_id(self):
        """Get review id."""
        return self.id

    def get_review(self):
        """Get review."""
        return self.review

    def get_predicted(self):
        """Get review predicted value."""
        return self.predicted

    def get_actual(self):
        """Get review actual value."""
        return self.actual


def get_review_by_id(review_id: int):
    """
    Get a review by its ID.

    Args:
        review_id (int): The ID of the review.

    Returns:
        Review: The review with the given ID.
    """
    filtered = list(filter(lambda x: x.id == review_id, reviews))
    if len(filtered) == 0:
        return None
    return filtered[0]


def get_model():
    """
    Load the model file.
    """
    return joblib.load("ml_models/c2_Classifier_Sentiment_Model")


def get_count_vectorizer():
    """
    Load the CountVectorizer file.
    """
    return joblib.load("ml_models/c1_BoW_Sentiment_Model.pkl")


def preprocess_review(review: str) -> np.ndarray:
    return mlSteps.preprocess_review(review)


def classify_review(review: str):
    """
    Makes a prediction based on the model and the input review.

    Args:
        review (str): The review to classify.

    Returns:
        int: The predicted sentiment label.
    """
    return mlSteps.classify_review(review)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Makes a sentiment prediction of the input message
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                review:
                    type: string
                    example: This is an example review.
    responses:
      200:
        description: The predicted class of the review
    """
    global NUMBER_OF_REQUESTS
    global NUMBER_OF_POSTIVE_PREDICTIONS
    global NUMBER_OF_NEGATIVE_PREDICTIONS

    # Increment the number of requests
    NUMBER_OF_REQUESTS += 1

    input: str = request.get_json().get("review")

    # Preprocess the review
    # logger.debug("Preprocessing review...")
    processed_review = preprocess_review(input)
    # logger.info("Preprocessing done.")
    # Make the prediction
    # logger.debug("Classifying review...")
    classification = classify_review(processed_review)
    # logger.info("Classification done.")

    predicted_class = int(classification[0])

    # Increment the number of positive or negative predictions
    if predicted_class == 1:
        NUMBER_OF_POSTIVE_PREDICTIONS += 1
    else:
        NUMBER_OF_NEGATIVE_PREDICTIONS += 1

    next_id = len(reviews)

    review = Review(next_id, input, predicted_class)
    reviews.append(review)
    return {"predicted_class": predicted_class, "review": review.__dict__}


@app.route("/checkPrediction", methods=["POST"])
@cross_origin()
def check_prediction():
    """
    Checks if the prediction is correct
    ---
    consumes:
      - application/json
    parameters:
        - name: prediction
          in: body
          description: The class that was predicted
          required: True
          schema:
            type: object
            properties:
                reviewId:
                    type: int
                    example: 1
                prediction_correct:
                    type: bool
                    example: false

    responses:
      200:
        description: the number of predictions that were correct and incorrect
      404:
        description: When the review with the given id does not exist
    """
    global NUMBER_OF_CORRECT_PREDICTIONS
    global NUMBER_OF_INCORRECT_PREDICTIONS

    review_id: int = request.get_json().get("reviewId")
    review: Review = get_review_by_id(review_id)

    if review is None:
        return "Review not found", 404

    prediction_correct: str = request.get_json().get("prediction_correct")

    if prediction_correct:
        NUMBER_OF_CORRECT_PREDICTIONS += 1
    else:
        NUMBER_OF_INCORRECT_PREDICTIONS += 1

    review.actual = (
        int(review.predicted)
        if prediction_correct
        else int(not review.predicted)
    )
    return {
        "number_of_correct_predictions": NUMBER_OF_CORRECT_PREDICTIONS,
        "number_of_incorrect_predictions": NUMBER_OF_INCORRECT_PREDICTIONS,
    }


@app.route("/getReviews", methods=["GET"])
@cross_origin()
def get_reviews():
    """
    Get a list with all reviews
    ---
    responses:
      200:
        description: A list with all predictions made
    """
    return {
        "reviews": [review.review for review in reviews],
    }


@app.route("/getPredictions", methods=["GET"])
@cross_origin()
def get_predictions():
    """
    Get a list with all predictions
    ---
    responses:
      200:
        description: A list with all predictions made
    """
    return {
        "reviews": [review.__dict__ for review in reviews],
    }


@app.route("/changeActual", methods=["POST"])
@cross_origin()
def change_actual():
    """
    Change the actual sentiment of a review
    ---
    consumes:
      - application/json
    parameters:
        - name: prediction
          in: body
          description: The class that was predicted
          required: True
          schema:
            type: object
            properties:
                reviewId:
                    type: int
                    example: 1
                actual:
                    type: int
                    example: 0

    responses:
      200:
        description: The updated review
      404:
        description: When the review with the given id does not exist
    """
    review_id: int = request.get_json().get("reviewId")
    actual: int = request.get_json().get("actual")
    review: Review = get_review_by_id(review_id)

    if review is None:
        return "Review not found", 404
    review.actual = actual
    return {"review": review.__dict__}, 200


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """
    Get the metrics for the model.

    Returns:
        Response: The metrics in Prometheus format.
    """
    # logger.info("Getting metrics...")
    global NUMBER_OF_REQUESTS
    global NUMBER_OF_POSTIVE_PREDICTIONS
    global NUMBER_OF_NEGATIVE_PREDICTIONS
    global NUMBER_OF_CORRECT_PREDICTIONS
    global NUMBER_OF_INCORRECT_PREDICTIONS

    message = "# HELP number_of_requests Number of requests\n"
    message += "# TYPE number_of_requests counter\n"

    message += "# HELP number_of_positive_predictions Number of positive predictions\n"
    message += "# TYPE number_of_positive_predictions counter\n"

    message += "# HELP number_of_negative_predictions Number of negative predictions\n"
    message += "# TYPE number_of_negative_predictions counter\n"

    message += (
        "# HELP number_of_correct_predictions Number of correct predictions\n"
    )
    message += "# TYPE number_of_correct_predictions counter\n"

    message += "# HELP number_of_incorrect_predictions Number of incorrect predictions\n"
    message += "# TYPE number_of_incorrect_predictions counter\n"

    message += f"number_of_requests {NUMBER_OF_REQUESTS}\n"
    message += (
        f"number_of_positive_predictions {NUMBER_OF_POSTIVE_PREDICTIONS}\n"
    )
    message += (
        f"number_of_negative_predictions {NUMBER_OF_NEGATIVE_PREDICTIONS}\n"
    )
    message += (
        f"number_of_correct_predictions {NUMBER_OF_CORRECT_PREDICTIONS}\n"
    )
    message += (
        "number_of_incorrect_predictions {NUMBER_OF_INCORRECT_PREDICTIONS}\n"
    )

    return Response(message, mimetype="text/plain")


app.run(host="0.0.0.0", port=8080, debug=True)
