<h1 align="center">🛡️ Toxic Comment Detector</h1>

<p align="center">
  <i>A lightweight NLP model that detects whether a comment is <b>toxic</b> or <b>non-toxic</b>.<br>
  Deployed with Gradio.
  (This is my first hand on exploration of NLP and deploying a project.)</i>
</p>

<p align="center">
  <a href="https://vinjin7-toxic-comment-detector.hf.space" target="_blank">
    🔗 <b>Live Demo on Hugging Face</b>
  </a>
</p>

<p align="center">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=py,pycharm" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ML-Logistic%20Regression-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/NLP-TF--IDF-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Deployed-brightgreen?style=flat-square"/>
   <a href="https://vinjin7-toxic-comment-detector.hf.space">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Space%20Running-brightgreen?style=flat-square"/>
  </a>
</p>





## 📸 Demo

<p align="center"> <b>•The Interface•</b></p>

![image](https://github.com/Jevinvy/toxic-comment-detector/blob/main/screenshots/Interface.png?raw=true)

<p align="center"> <b>• Toxic v/s Not Toxic •</b></p>
<p align="center">
  <img src="https://github.com/Jevinvy/toxic-comment-detector/blob/main/screenshots/example_toxic_1.png?raw=true" width="45%" />
  <img src="https://github.com/Jevinvy/toxic-comment-detector/blob/main/screenshots/example_not_toxic_2.png?raw=true" width="45%" />
</p>
<p align="center"> (Click on the image to see them fully) </p>

---

## 🔍 Project Overview

Toxic language online is both a technical and psychological challenge: it affects user safety, shapes social interactions, and influences how individuals perceive and respond to digital environments. This project approaches toxicity detection from a combined machine learning and cognitive science perspective, using computational models to approximate aspects of human judgment.

The system processes raw user comments, extracts linguistic patterns using TF-IDF vectorization, and predicts whether a message is toxic or non-toxic. TF-IDF highlights the importance of specific words or phrases, which parallels how humans rely on salient linguistic cues when evaluating emotional tone or harmful intent. A Logistic Regression classifier then uses these features to form a decision boundary—reflecting how simple cognitive models can separate categories based on learned evidence.

This project demonstrates end-to-end ML engineering:

- Cleaning and preparing noisy real-world text  
- Extracting interpretable features using TF-IDF  
- Training and evaluating a binary classifier  
- Understanding precision, recall, and model uncertainty  
- Saving and loading models for real-time inference  
- Designing a user-friendly Gradio interface  
- Deploying an interactive ML app on Hugging Face Spaces  

---

## 🧠 Model Architecture

### **TF-IDF Vectorizer**
The text is first transformed into numerical representations using TF-IDF, which highlights important words and down-weights common ones.

TF-IDF Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `max_features` | 50,000 | Only considers the top 50,000 most frequent terms. |
| `ngram_range` | (1, 2) | Considers single words (unigrams) and pairs of words (bigrams). |
| **Preprocessing** | English Stopwords Removed | Common, uninformative words are filtered out. | 

### **Classifier: Logistic Regression**
Logistic Regression is used as the classification model because it is:

- Fast and computationally lightweight
- Interpretable (weights can show which words influence toxicity)
- A strong baseline for text classification tasks

### **Pipeline Summary**
1. User enters a comment  
2. TF-IDF convert text into a sparse numeric vector 
3. Logistic Regression predicts toxicity  
4. Gradio UI displays:  
   - Toxic / Not Toxic  
   - Probability score  

---

## 📊 Dataset

The model is trained on the **Jigsaw Toxic Comment Classification** dataset, which consists of comments from Wikipedia's talk pages.

> 📝 **Note:** The dataset is proprietary and **not included** in this repository. You can obtain it from the Kaggle competition page: **[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)**.

### Binary Lable Creation
The original dataset conatins several toxicity categories:
- `toxic`
- `obscene`
- `insult`
- `identity hate`
- `threat`
- `severe_toxic`

These were merged into a single label:  
* `1` → **Toxic** (Any form of toxicity present)
* `0` → **Non-Toxic** (Clean comment) 

---

## 📁 Project Structure

```
toxic-comment-detector/
│
├── app.py                      # Main Gradio web app
├── requirements.txt            # 📦 Python dependencies (pandas, scikit-learn, gradio, etc.)
│
└── models/
    ├── baseline_model.joblib   # 🧠 The trained Logistic Regression classifier.
    └── tfidf_vectorizer.joblib # TF-IDF vectorizer
│
└── screenshots/                 
    ├── Interface.png            # Gradio interface image
    ├── example_not_toxic_1.png  # Example of not toxic
    ├── example_not_toxic_2.png
    ├── example_toxic_1.png      # Example of toxic
    └── example_toxic_2.png

```
## ▶️ Run Locally

To run the project on your machine:

```bash
pip install -r requirements.txt
python app.py
```
## Future Improvements

Potential futrther improvents:
- Upgrade to transformer models(BERT or DistilBERT) for higher accuracy
- Add explanation tools (highlighting words contributing to toxicity)
- Improve recall on subtle toxic comments
- Extend the model to multiple toxicity categories
- Add multilingual support

## ⚠️ DISCLAIMER

 The content of this repository is only for educational purpose and portfolio purpose only.
 It is a simple baseline classifier and may produce incorrect or biased predictions.
 It should not be used in production or real moderation decisions.
