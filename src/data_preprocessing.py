# src/data_preprocessing.py

import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer
import config


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    return df


def convert_to_hf_dataset(df):
    return Dataset.from_pandas(df)


def tokenize_dataset(dataset):
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    return tokenized, tokenizer
