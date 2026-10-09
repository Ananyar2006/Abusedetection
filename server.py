import logging
import os
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
  from deep_translator import GoogleTranslator

  TRANSLATOR_AVAILABLE = True
except ImportError:
  TRANSLATOR_AVAILABLE = False
  print("INFO: deep-translator not installed")

app = Flask(__name__, static_folder=".")
CORS(app, origins="*")

# ── Config ────────────────────────────────────────────────────────────────────
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
# Use a true multilingual XLM-RoBERTa toxicity model supporting all target languages
HF_MODEL = "unitary/multilingual-toxic-xlm-roberta"

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

# ── Language codes ─────────────────────────────────────────────────────────────
SARVAM_LANG_CODES = {
    "hindi": "hi-IN",
    "tamil": "ta-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
}
GOOGLE_LANG_CODES = {
    "hindi": "hi",
    "tamil": "ta",
    "kannada": "kn",
    "malayalam": "ml",
}


# ── Translation ────────────────────────────────────────────────────────────────
def translate_to_english(text: str, lang: str) -> str:
  # ── Sarvam API ─────────────────────────────────────────
  try:
    src_code = SARVAM_LANG_CODES.get(lang, "hi-IN")
    resp = requests.post(
        SARVAM_TRANSLATE_URL,
        headers={
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "input": text,
            "source_language_code": src_code,
            "target_language_code": "en-IN",
        },
        timeout=10,
    )
    data = resp.json()
    translated = data.get("translated_text") or data.get("translation") or ""
    if translated:
      return translated
  except Exception as e:
    print("Sarvam translate error:", e)

  # ── Fallback ───────────────────────────────────────────
  if TRANSLATOR_AVAILABLE:
    try:
      src_code = GOOGLE_LANG_CODES.get(lang, "auto")
      return GoogleTranslator(source=src_code, target="en").translate(text)
    except Exception as e:
      print("Fallback translate error:", e)

  return ""


# ── Inference (Hugging Face API) ──────────────────────────────────────────────
# ── Inference (Hugging Face API) ──────────────────────────────────────────────
# ── Inference (Hugging Face API - Model Driven) ──────────────────────────────
def predict(text: str, lang: str) -> dict:
  url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

  headers = {
      "Authorization": f"Bearer {HF_API_TOKEN}",
      "Content-Type": "application/json",
  }

  payload = {"inputs": text}

  try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    if not response.ok:
      print(f"Hugging Face HTTP {response.status_code}: {response.text}")
      raise RuntimeError(f"Hugging Face API error: {response.status_code}")

    data = response.json()

    # Safely extract score list from Hugging Face response structure
    scores_list = []
    if isinstance(data, list):
      if len(data) > 0 and isinstance(data[0], list):
        scores_list = data[0]
      else:
        scores_list = data
    elif isinstance(data, dict):
      if "labels" in data and "scores" in data:
        scores_list = [
            {"label": l, "score": s}
            for l, s in zip(data["labels"], data["scores"])
        ]

    # Aggregate scores across toxic, insult, obscene, and threat categories
    max_toxicity = 0.0
    non_toxic_score = 0.5

    for item in scores_list:
      lbl = str(item.get("label", "")).strip().lower()
      score = float(item.get("score", 0.0))

      # Check for negative/toxic classes
      if any(
          t in lbl for t in ["toxic", "insult", "obscene", "threat", "hate"]
      ):
        if score > max_toxicity:
          max_toxicity = score
      # Check for normal/non-toxic classes
      elif any(n in lbl for n in ["neutral", "normal", "non"]):
        non_toxic_score = score

    # Model evaluation threshold: If toxicity/insult probability is greater than 20%, flag as Offensive
    is_offensive = max_toxicity > 0.20

    label = "Offensive" if is_offensive else "Non-Offensive"
    label_id = 1 if is_offensive else 0

    # Calculate final percentage probabilities for the UI bars
    offensive_prob = round(max_toxicity * 100, 2)
    if not is_offensive and offensive_prob > 50.0:
      offensive_prob = round(100.0 - offensive_prob, 2)

    non_offensive_prob = round(100.0 - offensive_prob, 2)
    confidence = (
        max(offensive_prob, non_offensive_prob)
        if offensive_prob > 0
        else 95.0
    )

    return {
        "label": label,
        "label_id": label_id,
        "confidence": confidence,
        "language": lang,
        "text": text,
        "probs": {
            "non_offensive": non_offensive_prob,
            "offensive": offensive_prob,
        },
    }

  except Exception as e:
    print("Model prediction exception:", e)
    raise RuntimeError(f"Prediction failed: {str(e)}")
    
# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
  return send_from_directory(".", "index.html")


@app.route("/style.css")
def styles():
  return send_from_directory(".", "style.css")


@app.route("/health")
def health():
  return jsonify({
      "status": "ok",
      "hf_token_set": bool(HF_API_TOKEN),
      "sarvam_key_set": bool(SARVAM_API_KEY),
  })


# ── Analyze text ──────────────────────────────────────────────────────────────
@app.route("/analyze-text", methods=["POST"])
def analyze_text():
  data = request.get_json()

  text = data.get("text", "").strip()
  lang = data.get("language", "hindi").lower()

  if not text:
    return jsonify({"error": "Text is empty"}), 400

  result = predict(text, lang)
  result["translation"] = translate_to_english(text, lang)

  return jsonify(result)


# ── Analyze speech (transcript-based) ─────────────────────────────────────────
@app.route("/analyze-speech", methods=["POST"])
def analyze_speech():
  transcript = request.form.get("transcript", "").strip()
  lang = request.form.get("language", "hindi").lower()

  if not transcript:
    return jsonify({"error": "No transcript provided"}), 400

  result = predict(transcript, lang)
  result["transcript"] = transcript
  result["translation"] = translate_to_english(transcript, lang)

  return jsonify(result)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  print(f"Server running on port {port}")
  app.run(host="0.0.0.0", port=port)
