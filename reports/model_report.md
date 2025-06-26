# 📄 Model Report - Spam Mail Detection

## 🏗️ Model Architecture
- Pretrained Model: distilbert-base-uncased
- Classification Head: Dense Layer for Binary Classification

## ⚙️ Training Details
- Token Length: 95 tokens
- Batch Size: 8
- Learning Rate: 2e-5
- Epochs: 2
- Optimizer: AdamW with weight decay
- Mixed Precision: Enabled (fp16)

## 🔥 Improvements Made
- Enabled mixed precision to speed up training
- Applied weight decay to prevent overfitting
- Reset padding to max_length=95 to standardize input

## 📊 Key Observations
- High recall indicates good spam capture
- Slight precision trade-off due to aggressive spam detection
- Model generalizes well on validation set

## 🚀 Future Improvements
- Train with larger datasets for higher accuracy
- Explore models like RoBERTa or DeBERTa
- Add sender metadata or email headers as features
