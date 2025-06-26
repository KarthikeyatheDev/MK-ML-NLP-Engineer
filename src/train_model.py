# src/train_model.py

from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import config
from data_preprocessing import load_and_clean_data, convert_to_hf_dataset, tokenize_dataset
from model_utils import load_model, save_model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    df = load_and_clean_data("data/email_spam.csv")
    dataset = convert_to_hf_dataset(df).train_test_split(test_size=0.2, seed=42)
    tokenized_dataset, tokenizer = tokenize_dataset(dataset)

    model = load_model()

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=config.LOGGING_DIR,
        logging_steps=10,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()
    save_model(model, tokenizer)


if __name__ == "__main__":
    main()
