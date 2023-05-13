from flask import Flask, Response, request
from flask_cors import CORS, cross_origin
from flasgger import Swagger
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import nltk
import re
import joblib

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

app = Flask(__name__)
CORS(app)
swagger = Swagger(app)


"""Metrics"""
number_of_requests = 0
number_of_positive_predictions = 0
number_of_negative_predictions = 0


def get_model():
    """
    Load the model file.
    """
    return joblib.load('model/c2_Classifier_Sentiment_Model')
  
  
def get_count_vectorizer():
    """
    Load the CountVectorizer file.
    """
    return joblib.load('model/c1_BoW_Sentiment_Model.pkl')

def remove_stopwords(input: str) -> str:
  """
  Removes stopwords from the input string.

  Args:
      input (str): The string to remove stopwords from.

  Returns:
      str: The string without stopwords.
  """
  review = re.sub('[^a-zA-Z]', ' ', input)
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  result = ' '.join(review)
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
    cv = get_count_vectorizer()
    X = cv.transform([review]).toarray()
    return X
  
  
def classify_review(review: str):
    """
    Makes a prediction based on the model and the input review.

    Args:
        review (str): The review to classify.

    Returns:
        int: The predicted sentiment label.
    """
    model = get_model()
    result = model.predict(review)
    return result


@app.route('/', methods=['POST'])
@cross_origin()
def predict():
    """
    Make a hardcoded prediction
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
                msg:
                    type: string
                    example: This is an example msg.
    responses:
      200:
        description: Some result
    """
    global number_of_requests 
    global number_of_positive_predictions
    global number_of_negative_predictions
    
    # Increment the number of requests
    number_of_requests += 1
    
    msg: str = request.get_json().get('msg')
    
    # Preprocess the review
    review = preprocess_review(msg)
    # Make the prediction
    classification = classify_review(review)
    
    predicted_class = int(classification[0])
    
    # Increment the number of positive or negative predictions
    if predicted_class == 1:
        number_of_positive_predictions += 1
    else:
        number_of_negative_predictions += 1
    
    return {
        "predicted_class": predicted_class,
        "msg": msg
    }


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get the metrics for the model.

    Returns:
        Response: The metrics in Prometheus format.
    """
    global number_of_requests
    global number_of_positive_predictions
    global number_of_negative_predictions
    
    message = "# HELP number_of_requests Number of requests\n"
    message += "# TYPE number_of_requests counter\n"
    
    message += "# HELP number_of_positive_predictions Number of positive predictions\n"
    message += "# TYPE number_of_positive_predictions counter\n"
    
    message += "# HELP number_of_negative_predictions Number of neagative predictions\n"
    message += "# TYPE number_of_negative_predictions counter\n"

    message+= "number_of_requests{{page=\"index\"}} {}\n".format(number_of_requests)
    message+= "number_of_positive_predictions{{page=\"sub\"}} {}\n".format(number_of_positive_predictions)
    message+= "number_of_negative_predictions{{page=\"sub\"}} {}\n".format(number_of_negative_predictions)

    return Response(message, mimetype="text/plain")

app.run(host="0.0.0.0", port=8080, debug=True)
