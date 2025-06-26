# 📄 Submission - Spam Mail Detection using DistilBERT

---

## 🚀 Project Title
**Spam Mail Detection using DistilBERT**

---

## 👤 Author
**Karthikeya Mohan**  
Applying for **ML/NLP Engineer Internship**

---

## 🎯 Problem Statement
With the rise of phishing, scams, and fraudulent emails, spam detection is an essential problem in natural language processing. Manual email filtering is inefficient, error-prone, and non-scalable. The objective is to build a robust, transformer-based spam detection system capable of accurately classifying emails as **Spam** or **Ham (Not Spam)** based on their textual content.

---

## 🏗️ Approach and Workflow

### 🔥 Model Choice
- Selected **DistilBERT (distilbert-base-uncased)** — a lightweight version of BERT that retains ~97% of its performance while being 60% smaller and faster.
- Suitable for deployment scenarios with limited computational resources.

### 🔧 Frameworks and Tools
- **Hugging Face Transformers**
- **Hugging Face Datasets**
- **PyTorch**
- **Scikit-learn** (for metrics and visualizations)
- **Matplotlib** (for plots)

### 🛠️ Pipeline Steps
1. **Data Exploration**
   - Analyzed class distribution, null values, and text length distributions.
   - Visualized insights before processing.
2. **Data Preprocessing**
   - Removed null values and duplicates.
   - Converted labels to integer encoding.
3. **Tokenization**
   - Used Hugging Face tokenizer with truncation and padding.
   - `max_length = 95` chosen based on text length analysis.
4. **Model Fine-Tuning**
   - Hugging Face Trainer API used for ease and flexibility.
   - Hyperparameters:
     - Learning Rate: `2e-5`
     - Batch Size: `8`
     - Epochs: `5`
     - Weight Decay: `0.01`
5. **Evaluation**
   - Evaluated using:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
   - Confusion matrix generated for error analysis.
6. **Model Saving**
   - Saved model weights using `torch.save()` and tokenizer with `pickle` for compact storage.

---

## ⚙️ Model Details

| Hyperparameter | Value                  |
|----------------|-------------------------|
| Model          | distilbert-base-uncased |
| Batch Size     | 8                       |
| Learning Rate  | 2e-5                    |
| Epochs         | 5                       |
| Max Length     | 95                      |

---

## 📊 Evaluation Metrics

| Metric      | Value    |
|--------------|----------|
| Accuracy     | 97.4%    |
| Precision    | 98.36%   |
| Recall       | 96.95%   |
| F1-Score     | 97.65%   |
| Eval Loss    | 0.1497   |

✔️ The model generalizes very well with a high balance between precision and recall.

---

## 🚧 Challenges and Solutions

| Challenge                     | Solution                                                             |
|-------------------------------|----------------------------------------------------------------------|
| Training Time                  | Reduced dataset size to 5000 samples and used DistilBERT for speed. |
| Large Email Lengths            | Applied truncation with `max_length=95` to handle long emails.      |
| Overfitting Risk               | Applied weight decay and validation after every epoch.              |
| Model Size Too Large for GitHub| Saved model weights and tokenizer using `torch.save()` and `pickle`.|

---


## 💡 Key Learnings
- Hands-on understanding of fine-tuning transformer-based models for text classification tasks.
- Experience in handling real-world data issues like imbalance, text length, and noise.
- Gained practical skills in working with Hugging Face's Trainer API and PyTorch.
- Understood the importance of efficient model saving for storage-constrained environments.

---

## 🚀 Future Improvements
- Deploy the model as an API or Streamlit web application.
- Extend training on the full dataset for even better generalization.
- Incorporate email metadata (sender information, timestamps) for richer models.
- Experiment with more advanced models like RoBERTa or DeBERTa for further improvements.

---

## 🤝 Acknowledgements
- Hugging Face 🤗 for providing easy-to-use transformer tools.
- Kaggle for the email spam dataset.
- Chakaralya Analytics for the internship project challenge.
