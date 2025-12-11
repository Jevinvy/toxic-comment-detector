## ⚠️ Toxic Comment Detector 🗣️

An NLP project that classifies text as **toxic** or **non-toxic** using **TF-IDF features** and a **Logistic Regression model**. The project is deployed as an interactive Gradio web app so anyone can try the model in their browser.

| Live Demo | Source Code |
| :---: | :---: |
| 🔗 **[Hugging Face Spaces](https://vinjin7-toxic-comment-detector.hf.space)** | 💻 **[GitHub Repository](https://github.com/your-username/toxic-comment-detector)** |

---

## ✨ Project Overview

Online platforms often struggle with the issue of harmful or toxic comments. This project provides a **lightweight, baseline machine learning solution** to automatically detect toxicity in text.

This repository demonstrates key steps in a typical NLP project lifecycle:

* **Text Preprocessing** and **TF-IDF** (Term Frequency-Inverse Document Frequency) feature extraction.
* **Binary Classification** using a simple yet powerful **Logistic Regression** model.
* Model evaluation using metrics like precision, recall, and F1-score.
* Saving and loading production-ready models using `joblib`.
* **Real-time ML deployment** using the **Gradio** framework on **Hugging Face Spaces**.

---

## 📚 Dataset

The model is trained on the renowned **Jigsaw Toxic Comment Classification** dataset, which consists of comments from Wikipedia's talk pages.

> 📝 **Note:** The dataset is proprietary and **not included** in this repository. You can obtain it from the Kaggle competition page: **[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)**.

### Label Aggregation

The original dataset has multiple toxicity sub-labels (`toxic`, `obscene`, `insult`, `threat`, etc.). These were merged into a single **binary label** for simplification:

* `1` → **Toxic** (Any form of toxicity present)
* `0` → **Non-Toxic** (Clean comment)

---

## 🧠 Model & Feature Engineering

The core classifier is a **Logistic Regression** model, chosen for its speed and interpretability as a strong baseline for text classification.

The model is trained on features extracted using **TF-IDF Vectorization**.

### TF-IDF Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `max_features` | 50,000 | Only considers the top 50,000 most frequent terms. |
| `ngram_range` | (1, 2) | Considers single words (unigrams) and pairs of words (bigrams). |
| **Preprocessing** | English Stopwords Removed | Common, uninformative words are filtered out. |

---

## 🌐 Gradio Web Application

The interactive web app allows users to test the model in real-time without installing any code.

🔗 **Direct Link:** [https://vinjin7-toxic-comment-detector.hf.space](https://vinjin7-toxic-comment-detector.hf.space)

### Prediction Pipeline

When a user enters a comment into the app:

1.  The comment is tokenized and transformed using the pre-trained **`tfidf_vectorizer.joblib`**.
2.  The resulting feature vector is fed into the **`baseline_model.joblib`**.
3.  The model predicts the probability of the comment being **toxic** vs. **non-toxic**.
4.  The probability and final classification are displayed instantly.

---

## 📂 Project Structure

A clean layout for easy navigation:

```text
.
├── app.py                      # 🧑‍💻 The Gradio web interface and inference logic.
├── requirements.txt            # 📦 Python dependencies (pandas, scikit-learn, gradio, etc.).
└── models/
    ├── baseline_model.joblib   # 🧠 The trained Logistic Regression classifier.
    └── tfidf_vectorizer.joblib # ⚙️ The fitted TF-IDF feature extractor.
