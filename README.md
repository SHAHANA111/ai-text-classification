AI Text Classification (Spam Detection)
Project Overview

This project is a simple Machine Learning + Deep Learning based text classification system that detects whether an SMS message is Spam or Not Spam.

It uses:

TF-IDF Vectorization for text feature extraction
A simple Neural Network built with PyTorch
UCI SMS Spam Collection Dataset

Dataset:
Source: UCI Machine Learning Repository
Dataset: SMS Spam Collection
Contains labeled SMS messages:
0 → Not Spam
1 → Spam

Tech Stack:
Python
PyTorch
Pandas
NumPy
Scikit-learn

Workflow:
1. Load dataset
2. Clean text (lowercase, remove punctuation)
3. Convert text → numerical features using TF-IDF
4. Train Neural Network model
5. Evaluate using:
Accuracy
Confusion Matrix
Precision / Recall / F1-score
6. Save trained model
7. Test with custom inputs

Model Architecture:
Input Layer → TF-IDF features
Hidden Layer → Fully Connected (ReLU activation)
Output Layer → Binary classification (Spam / Not Spam)

Loss Function:

BCEWithLogitsLoss

Optimizer:

Adam
Evaluation Metrics
Accuracy: ~85% - 98% (varies per run)
Confusion Matrix used for detailed performance analysis
Precision & Recall used due to class imbalance

How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run training
python src/main.py

Project Structure
ai-text-classification/
│
├── src/
│   └── main.py
│
├── data/
│   └── processed/
│       └── spam.csv
│
├── models/
│   └── model.pt
│
├── requirements.txt
└── README.md

Future Improvements:
Use LSTM / Transformer models
Improve text preprocessing (lemmatization)
Deploy using FastAPI + React frontend
Add real-time email/SMS prediction API

Author:
Built as a learning project for NLP and Neural Networks.