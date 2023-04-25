# %%
# Save models for use in Flask app and Docker container
from transformers import pipeline
#%%
MODEL_VERSION = "1"
MODEL_PATH_EMOTION = "./saved-models/distilbert-emotion/" + MODEL_VERSION
MODEL_PATH_SENTIMENT = "./saved-models/distilbert-sentiment/" + MODEL_VERSION
# Load model for text classification by emotion:
# sadness, joy, fear, anger, surprise
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
emotion_model.save_pretrained(MODEL_PATH_EMOTION)
# Load model for text classification by broad sentiment only:
# positive/negative
sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
sentiment_model.save_pretrained(MODEL_PATH_SENTIMENT)