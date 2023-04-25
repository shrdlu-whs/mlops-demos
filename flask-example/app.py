# %%
from transformers import AutoTokenizer
from transformers import pipeline

# %%
MODEL_VERSION = "1"
MODEL_PATH_EMOTION = "./saved-models/distilbert-emotion/" + MODEL_VERSION
MODEL_PATH_SENTIMENT = "./saved-models/distilbert-sentiment/" + MODEL_VERSION
TOKENIZER = "distilbert-base-uncased"
#%%
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
# Load model for text classification by emotion:
# sadness, joy, fear, anger, surprise
emotion_model = pipeline("text-classification", model=MODEL_PATH_EMOTION, top_k=None)
# Load model for text classification by broad sentiment only:
# positive/negative
sentiment_model = pipeline("text-classification", model=MODEL_PATH_SENTIMENT)

# %%
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query', type=str, location='args')
# %%
# Returns max sentiment prediction and confidence for a given query text
class SentimentPrediction(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        sentiment_prediction = sentiment_model(user_query)
        sentiment = sentiment_prediction[0]['label'].lower()
        confidence = round(sentiment_prediction[0]['score'],4)

        # create JSON object
        output = {"sentiment": sentiment, "confidence": confidence}
        return output
# %%
# Returns max. predicted emotion and confidence for a given query text
class EmotionPrediction(Resource):
    def get(self):
        # Use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        # predict emotion
        emotion_prediction = emotion_model(user_query)
        max_prediction = max(emotion_prediction[0], key=lambda x:x['score'])
        emotion = max_prediction['label']
        confidence = round(max_prediction['score'],4)
        
        # Create JSON object
        output = {"emotion": emotion, "confidence": confidence}
        return output

#%%
# %%
# Returns all availanle models with description
class Models(Resource):
    def get(self):

        # create JSON object
        output = {"models":[{"model":"Sentiment Model","description":"Classifier detecting positive or negative overall sentiment of a text. "},{"model":"Emotion Model","description":"Classifier detecting the most likely emotion conveyed by a text."}]}
        
        return output

# %%
# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(Models, '/models')
api.add_resource(SentimentPrediction, '/models/sentiment')
api.add_resource(EmotionPrediction, '/models/emotion')


if __name__ == '__main__':
    app.run(debug=False)