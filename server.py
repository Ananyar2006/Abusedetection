import os
import time

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

# Models are tried in order; the first one that responds successfully is used.
# Override with env var HF_MODELS="modelA,modelB"
HF_MODELS = [
    m.strip()
    for m in os.environ.get(
        "HF_MODELS",
        "unitary/multilingual-toxic-xlm-roberta,"
        "unitary/toxic-bert,"
        "martin-ha/toxic-comment-model",
    ).split(",")
    if m.strip()
]

# Probability above which a text is labelled Offensive
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

# Remembers which model last worked so we don't retry broken ones every time
_working_model = None


# ── Translation ────────────────────────────────────────────────────────────────
def translate_to_english(text: str, lang: str) -> str:
  # Sarvam first
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

  # Fallback: Google
  if TRANSLATOR_AVAILABLE:
    try:
      src = GOOGLE_LANG_CODES.get(lang, "auto")
      return GoogleTranslator(source=src, target="en").translate(text)
    except Exception as e:
      print("Fallback translate error:", e)

  return ""


# ── Hugging Face helpers ──────────────────────────────────────────────────────
def call_hf(model: str, text: str, retries: int = 2):
  """Call one HF model. Returns parsed JSON or raises RuntimeError."""
  url = f"https://router.huggingface.co/hf-inference/models/{model}"
  headers = {
      "Authorization": f"Bearer {HF_API_TOKEN}",
      "Content-Type": "application/json",
      "X-Wait-For-Model": "true",
  }

  last_err = ""
  for attempt in range(retries + 1):
    resp = requests.post(
        url, headers=headers, json={"inputs": text}, timeout=45
    )

    if resp.status_code == 503:  # model loading
      last_err = f"503 loading: {resp.text[:150]}"
      time.sleep(5)
      continue

    if not resp.ok:
      raise RuntimeError(f"{model} -> HTTP {resp.status_code}: {resp.text[:200]}")

    try:
      data = resp.json()
    except Exception:
      raise RuntimeError(f"{model} -> non-JSON response: {resp.text[:200]}")

    if isinstance(data, dict) and "error" in data:
      raise RuntimeError(f"{model} -> {data['error']}")

    return data

  raise RuntimeError(f"{model} -> {last_err}")


def extract_offensive_score(data) -> float:
  """Turn the various HF classifier output shapes into one offensive prob."""
  # Normalise to a flat list of {"label":..., "score":...}
  if isinstance(data, list) and data and isinstance(data[0], list):
    items = data[0]
  elif isinstance(data, list):
    items = data
  else:
    items = []

  toxic_words = ["toxic", "insult", "obscene", "threat", "hate",
                 "offensive", "abusive", "abuse", "attack", "identity"]
  safe_words = ["non", "not", "neutral", "clean", "normal", "safe"]

  best = 0.0
  for item in items:
    if not isinstance(item, dict):
      continue
    label = str(item.get("label", "")).strip().lower()
    score = float(item.get("score", 0.0))

    if label in ("label_1", "1"):  # generic binary head: 1 = toxic
      best = max(best, score)
      continue
    if label in ("label_0", "0"):
      continue
    if any(label.startswith(s) or f"_{s}" in label or f"-{s}" in label
           for s in safe_words):
      continue  # e.g. "non-toxic", "not_offensive", "neutral"
    if any(t in label for t in toxic_words):
      best = max(best, score)

  return best


def score_text(text: str) -> dict:
  """Score text with the first working model. Raises if all models fail."""
  global _working_model

  order = ([_working_model] if _working_model else []) + [
      m for m in HF_MODELS if m != _working_model
  ]

  errors = []
  for model in order:
    try:
      data = call_hf(model, text)
      print(f"[{model}] RAW OUTPUT:", data)
      _working_model = model
      return {"model": model, "offensive": extract_offensive_score(data)}
    except Exception as e:
      print("HF error:", e)
      errors.append(str(e))
      if model == _working_model:
        _working_model = None

  raise RuntimeError(" | ".join(errors))


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(text: str, lang: str, translation: str = "") -> dict:
  # Score the original text and the English translation, keep the higher one
  results = [("original", score_text(text))]
  if translation and translation.strip().lower() != text.strip().lower():
    try:
      results.append(("translation", score_text(translation)))
    except Exception as e:
      print("Translation scoring failed:", e)

  source, best = max(results, key=lambda r: r[1]["offensive"])
  off_p = best["offensive"]
  is_offensive = off_p >= OFFENSIVE_THRESHOLD

  offensive_pct = round(off_p * 100, 2)
  non_offensive_pct = round(100 - offensive_pct, 2)

  return {
      "label": "Offensive" if is_offensive else "Non-Offensive",
      "label_id": 1 if is_offensive else 0,
      "confidence": max(offensive_pct, non_offensive_pct),
      "language": lang,
      "text": text,
      "model_used": best["model"],
      "scored_on": source,
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
      "hf_token_set": bool(HF_API_TOKEN),
      "sarvam_key_set": bool(SARVAM_API_KEY),
      "models": HF_MODELS,
      "active_model": _working_model,
      "threshold": OFFENSIVE_THRESHOLD,
  })


@app.route("/debug-model")
def debug_model():
  """Open /debug-model?text=you%20are%20stupid to see what each model returns."""
  text = request.args.get("text", "you are a stupid idiot")
  report = {}
  for model in HF_MODELS:
    try:
      data = call_hf(model, text, retries=1)
      report[model] = {
          "ok": True,
          "raw": data,
          "offensive_score": extract_offensive_score(data),
      }
    except Exception as e:
      report[model] = {"ok": False, "error": str(e)}
  return jsonify(report)


def run_analysis(text: str, lang: str, extra: dict = None):
  translation = translate_to_english(text, lang)
  try:
    result = predict(text, lang, translation)
  except Exception as e:
    return jsonify({
        "error": "Toxicity model unavailable",
        "details": str(e),
    }), 502
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
  print(f"Server running on port {port}")
  app.run(host="0.0.0.0", port=port)
