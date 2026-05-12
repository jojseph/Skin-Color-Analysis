# Agarthan Skin Tone Analyzer (ASTA) — Project Report

## What is this project?

The **Agarthan Skin Tone Analyzer (ASTA)** is a web application that determines a user's **seasonal colour type** — Spring, Summer, Autumn, or Winter — from a portrait photograph. Given a photo, the app returns a personalised colour palette, outfit recommendations, and makeup tips tailored to the user's skin tone and undertone.

## Purpose

The goal of ASTA is to make personal colour analysis accessible to anyone, without requiring a visit to a professional colour consultant. By leveraging a pre-trained Vision Language Model, the app can reason about skin undertone, depth, and colour harmony from a single photo — producing recommendations that would otherwise require expert human judgment.

## What is a Vision Language Model (VLM)?

A **Vision Language Model (VLM)** is a type of AI model that can process both images and text simultaneously. It is built on the **Transformer architecture** — the same foundation as large language models like GPT and Gemini — but extended to accept image input alongside text.

Images are broken into small fixed-size **patches** (e.g. 16×16 pixels), each patch is converted into a numerical vector, and those vectors are processed together with text tokens through the same attention mechanism. This allows the model to reason across both modalities at once — understanding the visual content of an image in the context of a text instruction.

VLMs are pre-trained on massive datasets of image–text pairs, giving them broad visual and language knowledge without requiring task-specific training data.

## What we used

| Component | Technology |
|---|---|
| **VLM** | Google Gemini 2.5 Flash (`gemini-2.5-flash`) |
| **VLM API** | Google AI API via the `google-genai` Python SDK |
| **Face detection** | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |
| **Image processing** | OpenCV, Pillow |
| **Web UI** | Streamlit |
| **Prompting technique** | Zero-shot structured (JSON-constrained) prompting |

## How it works

1. The user uploads a photo, uses their camera, or selects a sample image
2. OpenCV detects and crops the largest face, padding by 25% and resizing to 512×512 px
3. The cropped face is Base64-encoded and sent to Gemini 2.5 Flash with a structured prompt
4. Gemini analyses the face — skin undertone, depth, and colour harmony — and returns a JSON object
5. The app displays the season, colour palette, outfit tips, and makeup recommendations

Gemini is used **as-is** (pre-trained, zero-shot) — no additional fine-tuning was needed because the model already encodes colour theory and seasonal analysis knowledge from its pre-training data.
