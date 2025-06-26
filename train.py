# 🧠 Load Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# ⚙️ Training Arguments
training_args = TrainingArguments(
    output_dir="./models",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True
)


# 🚀 Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# 🔥 Train
trainer.train()

# ✅ Save Model
trainer.save_model("./models/spam_detector")
tokenizer.save_pretrained("./models/spam_detector")

print("✅ Training complete. Model saved in ./models/spam_detector")
