from flask import Flask, request
from flasgger import Swagger
import nltk
import re

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

app = Flask(__name__)
swagger = Swagger(app)

def preprocess(input: str) -> str:
  review = re.sub('[^a-zA-Z]', ' ', input)
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  result = ' '.join(review)
  return result

@app.route('/', methods=['POST'])
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
    review = preprocess(msg)
    return {
        "result": review,
    }

app.run(host="0.0.0.0", port=8080, debug=True)
