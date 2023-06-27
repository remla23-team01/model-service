"""
Module that contains the Flask application and the endpoints to serve the
model.
"""
import joblib
import numpy as np
from VersionMetrics import VersionMetrics
from flasgger import Swagger
from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# from logger.custom_logger import CustomFormatter
# import logging


# NLTK
nltk.download("stopwords")


ps = PorterStemmer()
all_stopwords = stopwords.words("english")
all_stopwords.remove("not")

# FLASK
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
versionMetrics: dict = {}

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


def remove_stopwords(input_str: str) -> str:
    """
    Removes stopwords from the input string.

    Args:
        input_str (str): The string to remove stopwords from.

    Returns:
        str: The string without stopwords.
    """
    #   logger.debug("Removing stopwords...")
    review = re.sub("[^a-zA-Z]", " ", input_str)
    review = review.lower()
    review = review.split()
    review = [
        ps.stem(word) for word in review if not word in set(all_stopwords)
    ]
    result = " ".join(review)
    #   logger.debug("Stopwords removed.")
    return result


def preprocess_review(review: str) -> np.ndarray:
    """
    Preprocesses the input review by removing stopwords and transforming it
    using the provided CountVectorizer.

    Args:
        review (str): The review to preprocess.

    Returns:
        np.ndarray: The preprocessed and transformed review.
    """
    review = remove_stopwords(review)
    # logger.debug("Loading CountVectorizer...")
    count_vec = get_count_vectorizer()
    # logger.info("CountVectorizer loaded.")
    transformed_view = count_vec.transform([review]).toarray()
    return transformed_view


def classify_review(review: str):
    """
    Makes a prediction based on the model and the input review.

    Args:
        review (str): The review to classify.

    Returns:
        int: The predicted sentiment label.
    """
    # logger.debug("Loading model...")
    model = get_model()
    # logger.info("Model loaded.")
    result = model.predict(review)
    return result

def get_version_metrics(version: str) -> VersionMetrics:

    if version not in versionMetrics:
        versionMetrics[version] = VersionMetrics()

    metrics = versionMetrics[version]
    return metrics

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
                version:
                    type: string
                    example: v1.0.0
    responses:
      200:
        description: The predicted class of the review
    """

    version: str = request.get_json().get("version")
    metrics: VersionMetrics = get_version_metrics(version)

    # Increment the number of requests
    metrics.number_of_requests += 1

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
        metrics.number_of_positive_predictions += 1
    else:
        metrics.number_of_negative_predictions += 1

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
                version:
                    type: string
                    example: v1.0.0

    responses:
      200:
        description: the number of predictions that were correct and incorrect
      404:
        description: When the review with the given id does not exist
    """
    version: str = request.get_json().get("version")
    metrics: VersionMetrics = get_version_metrics(version)
    review_id: int = request.get_json().get("reviewId")
    review: Review = get_review_by_id(review_id)

    if review is None:
        return "Review not found", 404

    prediction_correct: str = request.get_json().get("prediction_correct")

    if prediction_correct:
        metrics.number_of_correct_predictions += 1
    else:
        metrics.number_of_incorrect_predictions += 1

    review.actual = (
        int(review.predicted)
        if prediction_correct
        else int(not review.predicted)
    )
    return {
        # "number_of_correct_predictions": NUMBER_OF_CORRECT_PREDICTIONS,
        # "number_of_incorrect_predictions": NUMBER_OF_INCORRECT_PREDICTIONS,
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
    message = ""

    totalMetrics = VersionMetrics()

    for version in versionMetrics:
        metrics: VersionMetrics = versionMetrics[version]

        message += f"# HELP number_of_requests_{version} Number of requests of version {version}\n"
        message += f"# TYPE number_of_requests_{version} counter\n"
        message += f"number_of_requests_{version} {metrics.number_of_requests}\n\n"

        totalMetrics.number_of_requests += metrics.number_of_requests

        message += f"# HELP number_of_positive_predictions_{version} Number of positive predictions of version {version}\n"
        message += f"# TYPE number_of_positive_predictions_{version} counter\n"
        message += f"number_of_positive_predictions_{version} {metrics.number_of_positive_predictions}\n\n"
        totalMetrics.number_of_positive_predictions += metrics.number_of_positive_predictions

        message += f"# HELP number_of_negative_predictions_{version} Number of negative predictions of version {version}\n"
        message += f"# TYPE number_of_negative_predictions_{version} counter\n"
        message += f"number_of_negative_predictions_{version} {metrics.number_of_negative_predictions}\n\n"
        totalMetrics.number_of_negative_predictions += metrics.number_of_negative_predictions

        message += f"# HELP number_of_correct_predictions Number of correct predictions of version {version}\n"
        message += f"# TYPE number_of_correct_predictions_{version} counter\n"
        message += f"number_of_correct_predictions_{version} {metrics.number_of_correct_predictions}\n\n"
        totalMetrics.number_of_correct_predictions += metrics.number_of_correct_predictions

        message += f"# HELP number_of_incorrect_predictions_{version} Number of incorrect predictions of version {version}\n"
        message += f"# TYPE number_of_incorrect_predictions_{version} counter\n"
        message += f"number_of_incorrect_predictions_{version} {metrics.number_of_incorrect_predictions}\n\n"
        totalMetrics.number_of_incorrect_predictions += metrics.number_of_incorrect_predictions

    message += "# HELP number_of_requests Number of requests\n"
    message += "# TYPE number_of_requests counter\n"
    message += f"number_of_requests {totalMetrics.number_of_requests}\n\n"

    message += "# HELP number_of_positive_predictions Number of positive predictions\n"
    message += "# TYPE number_of_positive_predictions counter\n"
    message += f"number_of_positive_predictions {totalMetrics.number_of_positive_predictions}\n\n"

    message += "# HELP number_of_negative_predictions Number of negative predictions\n"
    message += "# TYPE number_of_negative_predictions counter\n"
    message += f"number_of_negative_predictions {totalMetrics.number_of_negative_predictions}\n\n"

    message += "# HELP number_of_correct_predictions Number of correct predictions\n"
    message += "# TYPE number_of_correct_predictions counter\n"
    message += f"number_of_correct_predictions {totalMetrics.number_of_correct_predictions}\n\n"

    message += "# HELP number_of_incorrect_predictions Number of incorrect predictions\n"
    message += "# TYPE number_of_incorrect_predictions counter\n"
    message += f"number_of_incorrect_predictions {totalMetrics.number_of_incorrect_predictions}\n\n"


    return Response(message, mimetype="text/plain")


app.run(host="0.0.0.0", port=8080, debug=True)
