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
.
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


---

## 🔗 Dataset Source
- **Kaggle:** [Email Spam Classification Dataset](https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)

Download the dataset manually and place it inside the `/data` folder.

---

## 🔬 Exploratory Data Analysis (EDA)
- ✔️ Checked class distribution (Spam vs. Ham)
- ✔️ Analyzed text length distributions
- ✔️ Identified and removed null values and duplicates

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 95.2%  |
| Precision | 94.8%  |
| Recall    | 96.1%  |
| F1-Score  | 95.4%  |

The model demonstrates high accuracy with a strong balance between precision and recall, indicating effective spam detection.

---

## 🚀 How to Run the Project

### ✅ Install Dependencies
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py --text "Congratulations! You have won a free iPhone. Claim now!"
