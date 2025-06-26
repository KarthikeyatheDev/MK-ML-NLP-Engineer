# src/model_utils.py

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import config


def load_model():
    return DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=2
    )


def save_model(model, tokenizer):
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.TOKENIZER_DIR)


def load_trained_model():
    model = DistilBertForSequenceClassification.from_pretrained(config.OUTPUT_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(config.TOKENIZER_DIR)
    return model, tokenizer


def predict(text, model, tokenizer):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LENGTH
    )
    with torch.no_grad():
        outputs = model(**tokens)
        preds = outputs.logits.argmax(dim=1).item()
        return "SPAM" if preds == 1 else "HAM"
