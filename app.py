import joblib
from pathlib import Path
import gradio as gr

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR/"models"

model = joblib.load(MODELS_DIR/"baseline_model.joblib")
vectorizer = joblib.load(MODELS_DIR/"tfidf_vectorizer.joblib")

def classify_comment(text: str):
    """
    Takes a text string and returns:
    - a label ("Toxic" / "Not Toxic"
    - the model's toxicity probability
    """

    text = text.strip()
    if not text:
        return "Please enter a comment", 0.0

    X = vectorizer.transform([text])

    if hasattr(model, "predict_proba"):
        proba_toxic = float(model.predict_proba(X)[0][1])
    else:
        pred = int(model.predict(X)[0])
        proba_toxic = float(pred)

    if proba_toxic >= 0.5:
        label = "😑 Toxic"
    else:
        label = "😊 Not Toxic"

    return label, round(proba_toxic, 3)

demo = gr.Interface(fn=classify_comment,
                    inputs=gr.Textbox(
                        label="Enter a comment",
                        lines=3,
                        placeholder="Type something here..."),
                    outputs=[gr.Textbox(label="Prediction"),
                             gr.Slider(0, 1, label="Toxicity probability", interactive=False)],
                    title="Toxic Comment Detector",
                    description="Type a comment and the model will predict whether it's toxic or not",
                    )
if __name__ == "__main__":
    demo.launch()
