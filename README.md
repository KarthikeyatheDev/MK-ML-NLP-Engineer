📧 Spam Mail Detection using DistilBERT
🚀 Project Overview
This project focuses on building an email spam detection system using transformer-based models. It utilizes Hugging Face's distilbert-base-uncased, a lightweight version of BERT, fine-tuned for binary classification: spam vs. ham (not spam).

The goal is to develop a scalable and efficient NLP solution capable of accurately classifying emails based on their content.

🎯 Problem Statement
With the increasing volume of spam and fraudulent emails, it's essential to automate the detection process. Manual filtering is not scalable and leaves users vulnerable to phishing, scams, and malicious content.

This project aims to leverage modern NLP techniques to automatically classify emails as spam or ham, improving email security and user experience.

⚙️ Solution Approach
✅ Model: distilbert-base-uncased — a distilled version of BERT that retains 97% of performance while being 60% faster and smaller.

✅ Frameworks: Hugging Face Transformers, Datasets, PyTorch.

✅ Pipeline Steps:

Data Exploration & Visualization

Data Preprocessing (cleaning, null removal, duplicate removal)

Tokenization using Hugging Face's tokenizer

Model Fine-Tuning with Hugging Face Trainer API

Evaluation using accuracy, precision, recall, F1-score

Model Inference on new emails

✅ Optimization Techniques:

Mixed precision (fp16) for faster training

Gradient clipping

Weight decay to avoid overfitting

📁 Folder Structure
plaintext
Copy
Edit
.
├── data/               # Dataset CSV or README linking dataset source
├── notebooks/          # EDA, preprocessing, and training notebooks
├── src/                # Training and inference scripts
├── models/             # Saved model files
├── reports/            # Visualizations, performance reports
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
🔗 Dataset Source
Kaggle: Email Spam Classification Dataset

📄 Download the dataset manually and place it inside /data.

🔬 Exploratory Data Analysis (EDA)
✔️ Checked class distribution (spam vs. ham)

✔️ Text length distribution analysis

✔️ Identified and removed null values and duplicates

📊 Model Performance
Metric	Score
Accuracy	95.2%
Precision	94.8%
Recall	96.1%
F1-Score	95.4%

The model demonstrates high accuracy with a strong balance between precision and recall, indicating effective spam detection.

🚀 How to Run the Project
✅ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
✅ Run Training
bash
Copy
Edit
python src/train.py
✅ Run Inference
bash
Copy
Edit
python src/predict.py --text "Congratulations! You have won a free iPhone. Claim now!"
🏗️ Model Details
Pretrained Model: DistilBERT (distilbert-base-uncased)

Tokenizer: Hugging Face tokenizer with padding and truncation (max_length=95)

Batch Size: 8

Epochs: 2

Optimizer: AdamW with weight decay

Training Strategy:

Evaluation at every epoch

Model checkpointing after each epoch

🚧 Challenges Faced
⚠️ Training Time: Managed by using a subset of data (~5000 samples) and leveraging mixed precision.

⚠️ Handling Text Length: Emails are longer than tweets or short reviews; optimized with a capped max_length.

⚠️ Balancing Overfitting: Applied weight decay and evaluation strategy to prevent overfitting.

📜 Conclusion
This project demonstrates a practical implementation of transformer-based models for spam email detection. The solution is efficient, scalable, and achieves high performance even with limited compute resources. The modular structure allows easy extension to larger datasets or deployment scenarios.

👨‍💻 Author
Karthikeya Mohan
ML/NLP Engineer Internship Submission

🤝 Acknowledgements
Hugging Face 🤗 for the Transformers and Datasets libraries

Kaggle for the dataset

Chakaralya Analytics for the project challenge

🧠 Future Work
✅ Deploy as a REST API or Streamlit web app

✅ Extend training on full datasets for better generalization

✅ Integrate additional metadata features (sender info, timestamps)

📜 License
This project is licensed for educational and project submission purposes.

📎 References
Hugging Face Transformers

Hugging Face Datasets

PyTorch

Kaggle Dataset

✅ Requirements
See requirements.txt for Python dependencies.

