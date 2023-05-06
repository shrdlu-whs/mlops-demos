import os
# %%
import torch
from tokenizers import Tokenizer
# %%
MODEL_VERSION = "1"
MODEL_PATH_EMOTION = "./saved-models/distilbert-emotion/" + MODEL_VERSION
MODEL_PATH_SENTIMENT = "./saved-models/distilbert-sentiment/" + MODEL_VERSION
TOKENIZER = "distilbert-base-uncased"
SEQUENCE_LENGTH = 64
#%%
tokenizer = Tokenizer.from_pretrained(TOKENIZER)
sentiment_labels = ['negative','positive']
emotion_labels = ['sadness','joy', 'love', 'anger', 'fear', 'surprise']
#%%
def load_model(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = torch.jit.load(os.path.join(model_dir, "model.pt"))
    print("model loaded from "+ model_dir)
    
    return loaded_model.to(device)

def encode_input(data):
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            pass
        else:
            raise ValueError("Unsupported input type. Input type can be a string or an non-empty list. \
                             I got {}".format(data))
        encoded_sentences = [tokenizer.encode(x, add_special_tokens=True) for x in data]
        # pad shorter sentence
        padded =  torch.zeros(len(encoded_sentences), SEQUENCE_LENGTH) 
        for i, s in enumerate(encoded_sentences):
            padded[i, :len(s.ids)] = torch.tensor(s.ids)
     
        # create mask
        mask = (padded != 0)

        return padded.long(), mask.long()

import torch.nn.functional as F
def predict(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_id, input_mask = input_data
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)
    
    # Do inference
    with torch.no_grad():
        y = model(input_id, attention_mask=input_mask)[0]
    probabilities = F.softmax(y, dim=-1)
    return probabilities

#%%
# Load model for text classification by emotion:
# sadness, joy, fear, anger, surprise
emotion_model = load_model(MODEL_PATH_EMOTION)
# Load model for text classification by broad sentiment only:
# positive/negative
sentiment_model = load_model(MODEL_PATH_SENTIMENT)
#%%
from flask import Flask
from flask_restful import reqparse, Api, Resource

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query', type=str, location='args')
# %%
# Returns max sentiment prediction and confidence for a given query text
class SentimentPrediction(Resource):
    def get(self):
        # use parser and find the users query
        args = parser.parse_args()
        user_query = args['query']
        input_data = encode_input(user_query)
        sentiment_prediction = predict(input_data, sentiment_model)[0]
        sentiment_idx = torch.argmax(sentiment_prediction)
        sentiment_prediction = sentiment_prediction.detach().cpu().numpy()
        sentiment = sentiment_labels[sentiment_idx]
        confidence = float(round(sentiment_prediction[sentiment_idx],4))

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
        input_data = encode_input(user_query)
        # predict emotion
        emotion_prediction = predict(input_data, emotion_model)[0]
        emotion_idx = torch.argmax(emotion_prediction)
        emotion_prediction = emotion_prediction.detach().cpu().numpy()
        emotion = emotion_labels[emotion_idx]
        confidence = float(round(emotion_prediction[emotion_idx],4))

        # create JSON object
        output = {"emotion": emotion, "confidence": confidence}
        return output

#%%
# %%
# Returns all available models with description
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