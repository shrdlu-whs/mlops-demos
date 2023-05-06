# %%
# Save models for use in Flask app and Docker container
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
#%%
MODEL_VERSION = "1"
MODEL_PATH_EMOTION = os.path.join("./saved-models/distilbert-emotion/", MODEL_VERSION)
MODEL_PATH_SENTIMENT = os.path.join("./saved-models/distilbert-sentiment/", MODEL_VERSION)
SEQUENCE_LENGTH = 64
JIT_EXPORT = True

device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load model for text classification by emotion:
# sadness, joy, fear, anger, surprise
emotion_model = DistilBertForSequenceClassification.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion", top_k=None, torchscript=True)
# Load model for text classification by broad sentiment only:
# positive/negative
sentiment_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", torchscript=True)
# %%
# Save Jit model trace
if JIT_EXPORT:
    # Create dummy input for trace
    for_jit_trace_input_ids = [0] * SEQUENCE_LENGTH
    for_jit_trace_attention_masks = [0] * SEQUENCE_LENGTH
    for_jit_trace_input = torch.tensor([for_jit_trace_input_ids])
    for_jit_trace_masks = torch.tensor([for_jit_trace_input_ids])

    traced_emotion_model = torch.jit.trace(
        emotion_model, [for_jit_trace_input.to(device), for_jit_trace_masks.to(device)]
    )
    torch.jit.save(traced_emotion_model, os.path.join(MODEL_PATH_EMOTION, "model.pt"))

    traced_sentiment_model = torch.jit.trace(
        sentiment_model, [for_jit_trace_input.to(device), for_jit_trace_masks.to(device)]
    )
    torch.jit.save(traced_sentiment_model, os.path.join(MODEL_PATH_SENTIMENT, "model.pt"))
