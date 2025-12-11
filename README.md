# Toxic Comment Detector

An NLP project that classifies text as **toxic** or **non-toxic** using TF-IDF features and a Logistic Regression model. The project is deployed as an interactive Gradio web app so anyone can try the model in their browser.

🔗 **Live Demo:** https://vinjin7-toxic-comment-detector.hf.space  

---

## 🔍 Overview

Online platforms often have to deal with toxic or harmful comments.
This project builds a lightweight machine learning model that automatically detects whether a comment is toxic.

It demonstrates:

- Text preprocessing and TF-IDF feature extraction  
- Binary classification (Logistic Regression)  
- Model evaluation (precision, recall, F1-score)  
- Saving/loading models with joblib  
- Deploying a real-time ML app using Gradio + Hugging Face Spaces  

---

## 📊 Dataset

The model is trained on the **Jigsaw Toxic Comment Classification** dataset (Wikipedia comments).  
The dataset is copyrighted, so it is **not included** in this repository.

You can find it on Kaggle:
> Jigsaw Toxic Comment Classification Challenge

Multiple labels (`toxic`, `obscene`, `insult`, etc.) are merged into one **binary label**:

- `1` → toxic  
- `0` → non-toxic  

---

## 🧠 Model

The classifier is a **Logistic Regression** model trained on TF-IDF features.

**TF-IDF settings**:

- `max_features = 50_000`
- `ngram_range = (1, 2)`
- English stopwords removed

This simple baseline model performs well for text classification tasks.

---

## 🌐 Web App

Built using **Gradio** and deployed on **Hugging Face Spaces**.

🔗 **Try the app here:** https://vinjin7-toxic-comment-detector.hf.space  

When a user enters a comment:

1. The text is transformed using TF-IDF  
2. The Logistic Regression model predicts toxic vs non-toxic  
3. The toxicity probability is displayed  

---

## 🗂 Project Structure

```text
.
├── app.py                # Gradio web app
├── requirements.txt      # Dependencies for deployment
├── models/
│   ├── baseline_model.joblib         # Trained classifier
│   └── tfidf_vectorizer.joblib       # Vectorizer used for feature extraction
└── (optional: training scripts, screenshots)
