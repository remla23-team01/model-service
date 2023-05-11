from flask import Flask, request
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
    
    msg: str = request.get_json().get('msg')
    
    # Preprocess the review
    review = preprocess_review(msg)
    # Make the prediction
    classification = classify_review(review)
    
    return {
        "predicted_class": int(classification[0]),
        "msg": msg
    }

app.run(host="0.0.0.0", port=8080, debug=True)
