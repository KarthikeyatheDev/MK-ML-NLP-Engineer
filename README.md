# 📧 Spam Mail Detection using DistilBERT

## 🚀 Project Overview
This project focuses on building an **email spam detection system** using transformer-based models. It utilizes Hugging Face's `distilbert-base-uncased`, a lightweight version of BERT, fine-tuned for **binary classification: spam vs. ham (not spam)**.

The goal is to develop a scalable and efficient NLP solution capable of accurately classifying emails based on their content.

---

## 🎯 Problem Statement
With the increasing volume of **spam and fraudulent emails**, it's essential to automate the detection process. Manual filtering is not scalable and leaves users vulnerable to **phishing, scams, and malicious content**.

This project leverages **modern NLP techniques** to automatically classify emails as spam or ham, improving email security and user experience.

---

## ⚙️ Solution Approach

### ✅ Model
- `distilbert-base-uncased` — a distilled version of BERT that retains **97% of performance** while being **60% faster and smaller**.

### ✅ Frameworks
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch

### ✅ Pipeline Steps
- Data Exploration & Visualization
- Data Preprocessing (null removal, duplicate removal, cleaning)
- Tokenization using Hugging Face tokenizer
- Model Fine-Tuning with Hugging Face Trainer API
- Evaluation using Accuracy, Precision, Recall, F1-Score
- Model Inference on new email samples

### ✅ Optimization Techniques
- Mixed Precision (`fp16`) for faster training
- Gradient Clipping
- Weight Decay to prevent overfitting

---

## 📁 Folder Structure

├── data/ # Dataset CSV or source reference

├── notebooks/ # EDA, preprocessing, and training notebooks

├── src/ # Training and inference scripts

├── models/ # Saved model files and tokenizer

├── reports/ # Visualizations, reports, metrics

├── README.md # Project documentation

├── requirements.txt # Python dependencies

├── submission.md # Project summary for submission

├── train.py # Script to run training


---

## 🔗 Dataset Source
- **Kaggle:** [Email Spam Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)

---

## 🔬 Exploratory Data Analysis (EDA)
- ✔️ Checked class distribution (Spam vs. Ham)
- ✔️ Analyzed text length distributions
- ✔️ Identified and removed null values and duplicates

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 97.4%  |
| Precision | 98.36%  |
| Recall    | 96.95%  |
| F1-Score  | 97.65%  |

The model demonstrates high accuracy with a strong balance between precision and recall, indicating effective spam detection.

---

## 🚀 How to Run the Project

### ✅ Install Dependencies

pip install -r requirements.txt

python src/train.py

python src/predict.py --text "Congratulations! You have won a free iPhone. Claim now!"

---

## 🏗️ Model Details
Pretrained Model: DistilBERT (distilbert-base-uncased)

Tokenizer: Hugging Face tokenizer with padding and truncation (max_length=95)

Batch Size: 8

Epochs: 5

Optimizer: AdamW with weight decay

Training Strategy:

Evaluation at every epoch

Checkpoint saving after each epoch

---

## 🚧 Challenges Faced
⚠️ Training Time: Managed by using a subset (~5000 samples) and enabling mixed precision

⚠️ Handling Text Length: Emails are longer than tweets or short reviews; optimized with max_length=95

⚠️ Preventing Overfitting: Applied weight decay and evaluation at each epoch

---

## 📜 Conclusion
This project demonstrates a practical implementation of transformer-based models for spam email detection. The solution is efficient, scalable, and achieves high performance even with limited compute resources.The modular structure allows easy extension to larger datasets or deployment scenarios.

---

## 👨‍💻 Author
Karthikeya Mohan

ML/NLP Engineer Internship Submission

---

## 🤝 Acknowledgements
Hugging Face 🤗 for Transformers and Datasets

Kaggle for the dataset

Chakaralya Analytics for the project challenge

## 🧠 Future Work
✅ Deploy as a REST API or Streamlit web app

✅ Extend training on the full dataset for better generalization

✅ Integrate metadata features (sender info, timestamps)

---

## 📜 License

This project is licensed for educational and project submission purposes.
