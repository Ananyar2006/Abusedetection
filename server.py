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
# A real toxicity model that runs inside this server (downloaded once, ~440 MB).
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "unitary/toxic-bert")
OFFENSIVE_THRESHOLD = float(os.environ.get("OFFENSIVE_THRESHOLD", "0.5"))

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

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
  if SARVAM_API_KEY:
    try:
      resp = requests.post(
          SARVAM_TRANSLATE_URL,
          headers={
              "api-subscription-key": SARVAM_API_KEY,
              "Content-Type": "application/json",
          },
          json={
              "input": text,
              "source_language_code": SARVAM_LANG_CODES.get(lang, "hi-IN"),
              "target_language_code": "en-IN",
          },
          timeout=10,
      )
      if resp.ok:
        data = resp.json()
        translated = data.get("translated_text") or data.get("translation")
        if translated:
          return translated
      else:
        print(f"Sarvam HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
      print("Sarvam translate error:", e)

  if TRANSLATOR_AVAILABLE:
    try:
      src = GOOGLE_LANG_CODES.get(lang, "auto")
      return GoogleTranslator(source=src, target="en").translate(text)
    except Exception as e:
      print("Fallback translate error:", e)

  return ""


# ── Local toxicity model ──────────────────────────────────────────────────────
_classifier = None


def get_classifier():
  """Load the model once, on first use."""
  global _classifier
  if _classifier is None:
    from transformers import pipeline

    print(f"Loading model {LOCAL_MODEL} (first time may take a minute)...")
    _classifier = pipeline(
        "text-classification",
        model=LOCAL_MODEL,
        top_k=None,  # return the score of every label
        truncation=True,
        max_length=256,
    )
    print("Model loaded.")
  return _classifier


def offensive_score(text: str) -> float:
  """Highest toxicity-type probability (toxic, insult, obscene, threat...)."""
  out = get_classifier()(text)
  if out and isinstance(out[0], list):
    out = out[0]
  print(f"MODEL OUTPUT for {text!r}:", out)

  best = 0.0
  for item in out:
    label = str(item["label"]).lower()
    if label in ("label_0", "non-toxic", "not_toxic", "neutral"):
      continue
    best = max(best, float(item["score"]))
  return best


def predict(text: str, lang: str, translation: str) -> dict:
  # The model is English, so judge the English translation when we have one.
  scored_text = translation if translation else text
  off = offensive_score(scored_text)

  is_offensive = off >= OFFENSIVE_THRESHOLD
  offensive_pct = round(off * 100, 2)
  non_offensive_pct = round(100 - offensive_pct, 2)

  return {
      "label": "Offensive" if is_offensive else "Non-Offensive",
      "label_id": 1 if is_offensive else 0,
      "confidence": max(offensive_pct, non_offensive_pct),
      "language": lang,
      "text": text,
      "model_used": LOCAL_MODEL,
      "scored_text": scored_text,
      "probs": {
          "non_offensive": non_offensive_pct,
          "offensive": offensive_pct,
      },
  }


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
      "sarvam_key_set": bool(SARVAM_API_KEY),
      "model": LOCAL_MODEL,
      "model_loaded": _classifier is not None,
      "threshold": OFFENSIVE_THRESHOLD,
  })


def run_analysis(text: str, lang: str, extra: dict = None):
  translation = translate_to_english(text, lang)
  try:
    result = predict(text, lang, translation)
  except Exception as e:
    print("Prediction error:", e)
    return jsonify({"error": "Model error", "details": str(e)}), 500
  result["translation"] = translation
  if extra:
    result.update(extra)
  return jsonify(result)


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
  data = request.get_json(silent=True) or {}
  text = data.get("text", "").strip()
  lang = data.get("language", "hindi").lower()

  if not text:
    return jsonify({"error": "Text is empty"}), 400
  return run_analysis(text, lang)


@app.route("/analyze-speech", methods=["POST"])
def analyze_speech():
  transcript = request.form.get("transcript", "").strip()
  lang = request.form.get("language", "hindi").lower()

  if not transcript:
    return jsonify({"error": "No transcript provided"}), 400
  return run_analysis(transcript, lang, {"transcript": transcript})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  get_classifier()  # load the model now so the first request is fast
  print(f"Server running on port {port}")
  app.run(host="0.0.0.0", port=port)
