import gradio as gr
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

def predict(image):
    predictions = pipeline(image)
    return {p["label"]: p["score"] for p in predictions}

gr.Interface(
    predict,
    inputs=gr.Image(label="Upload hot dog candidate", type="filepath"),
    outputs=gr.Label(num_top_classes=2),
    title="Hot Dog? Or Not?",
).launch()