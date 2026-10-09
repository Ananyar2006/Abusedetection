import logging
import os
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from transformers import pipeline

try:
  from deep_translator import GoogleTranslator

  TRANSLATOR_AVAILABLE = True
except ImportError:
  TRANSLATOR_AVAILABLE = False


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


# --------------------------------------------------
# MODEL CONFIGURATION (XLM-ROBERTA LOCAL PIPELINE)
# --------------------------------------------------

# Replace with your local folder path or fine-tuned xlm-roberta model if applicable
HF_MODEL = "xlm-roberta-base"

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "").strip()
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

# Initialize local text-classification pipeline globally
app.logger.info("Loading local XLM-RoBERTa pipeline (%s)...", HF_MODEL)
try:
  classifier = pipeline("text-classification", model=HF_MODEL, top_k=None)
  app.logger.info("Local XLM-RoBERTa pipeline loaded successfully!")
except Exception as e:
  app.logger.error("Failed to load local pipeline: %s", e)
  classifier = None


# --------------------------------------------------
# SHARED HTTP SESSION
# --------------------------------------------------

session = requests.Session()


# --------------------------------------------------
# TRANSLATION
# --------------------------------------------------


def translate_to_english(text, lang):
  if not text:
    return ""

  lang = lang.lower().strip()
  source_code = SARVAM_LANG_CODES.get(lang)

  if not source_code:
    return ""

  if SARVAM_API_KEY:
    try:
      response = session.post(
          SARVAM_TRANSLATE_URL,
          headers={
              "api-subscription-key": SARVAM_API_KEY,
              "Content-Type": "application/json",
          },
          json={
              "input": text[:2000],
              "source_language_code": source_code,
              "target_language_code": "en-IN",
          },
          timeout=20,
      )

      if response.ok:
        result = response.json()
        translated = result.get("translated_text", "")
        if translated:
          return translated

    except (requests.RequestException, ValueError) as exc:
      app.logger.warning("Sarvam translation failed: %s", exc)

  if TRANSLATOR_AVAILABLE:
    try:
      return GoogleTranslator(
          source=GOOGLE_LANG_CODES[lang],
          target="en",
      ).translate(text)
    except Exception as exc:
      app.logger.warning("Google translation failed: %s", exc)

  return ""


# --------------------------------------------------
# XLM-ROBERTA CLASSIFICATION
# --------------------------------------------------


def predict(text, lang):
  if classifier is None:
    raise RuntimeError("Classification model is not loaded on the server.")

  try:
    # Run text classification locally using XLM-RoBERTa
    output = classifier(text)

    # Output structure from top_k=None is typically a list of dicts: [[{'label': '...', 'score': ...}, ...]]
    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], list):
      scores_list = output[0]
    else:
      scores_list = output

    score_map = {
        str(item["label"]).strip().lower(): float(item["score"])
        for item in scores_list
    }

    # Map model labels to your application labels (adjust keys if your fine-tuned model uses different label names like LABEL_0/LABEL_1)
    offensive_score = score_map.get("offensive", score_map.get("label_1", 0.0))
    non_offensive_score = score_map.get(
        "non-offensive", score_map.get("non_offensive", score_map.get("label_0", 0.0))
    )

    # Fallback if binary mapping names differ
    if offensive_score == 0.0 and non_offensive_score == 0.0 and len(scores_list) >= 2:
      offensive_score = scores_list[1]["score"]
      non_offensive_score = scores_list[0]["score"]

    label = (
        "Offensive" if offensive_score > non_offensive_score else "Non-Offensive"
    )
    confidence = max(offensive_score, non_offensive_score) * 100

    return {
        "label": label,
        "label_id": 1 if label == "Offensive" else 0,
        "confidence": round(confidence, 2),
        "language": lang,
        "text": text,
        "probs": {
            "non_offensive": round(non_offensive_score * 100, 2),
            "offensive": round(offensive_score * 100, 2),
        },
    }

  except Exception as exc:
    app.logger.error("Classification error: %s", exc)
    raise RuntimeError(f"Classification failed: {str(exc)}")


# --------------------------------------------------
# ROUTES
# --------------------------------------------------


@app.route("/", methods=["GET"])
def index():
  return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css", methods=["GET"])
def styles():
  return send_from_directory(BASE_DIR, "style.css")


@app.route("/health", methods=["GET"])
def health():
  return jsonify({
      "status": "ok",
      "model_loaded": classifier is not None,
      "sarvam_key_set": bool(SARVAM_API_KEY),
      "translator_fallback_available": TRANSLATOR_AVAILABLE,
  })


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
  data = request.get_json(silent=True)

  if not isinstance(data, dict):
    return jsonify({"error": "Send a valid JSON request."}), 400

  text = data.get("text", "")
  lang = str(data.get("language", "hindi")).lower().strip()

  if not isinstance(text, str) or not text.strip():
    return jsonify({"error": "Please enter some text."}), 400

  text = text.strip()

  if len(text) > 5000:
    return jsonify({"error": "Text must be 5000 characters or fewer."}), 400

  if lang not in SARVAM_LANG_CODES:
    return jsonify({
        "error": (
            "Unsupported language. Choose Hindi, Tamil, Kannada or"
            " Malayalam."
        )
    }), 400

  try:
    result = predict(text, lang)
    result["translation"] = translate_to_english(text, lang)
    return jsonify(result)

  except RuntimeError as exc:
    app.logger.error("Text analysis failed: %s", exc)
    return jsonify({"error": str(exc)}), 502

  except Exception:
    app.logger.exception("Unexpected text analysis error")
    return jsonify({"error": "An unexpected server error occurred."}), 500


@app.route("/analyze-speech", methods=["POST"])
def analyze_speech():
  transcript = request.form.get("transcript", "").strip()
  lang = request.form.get("language", "hindi").lower().strip()

  if not transcript:
    return jsonify({"error": "No speech transcript was provided."}), 400

  if lang not in SARVAM_LANG_CODES:
    return jsonify({"error": "Unsupported language."}), 400

  try:
    result = predict(transcript, lang)
    result["transcript"] = transcript
    result["translation"] = translate_to_english(transcript, lang)

    return jsonify(result)

  except RuntimeError as exc:
    app.logger.error("Speech analysis failed: %s", exc)
    return jsonify({"error": str(exc)}), 502

  except Exception:
    app.logger.exception("Unexpected speech analysis error")
    return jsonify({"error": "An unexpected server error occurred."}), 500


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------


@app.errorhandler(413)
def request_too_large(_error):
  return jsonify({"error": "The request is too large."}), 413


@app.errorhandler(404)
def not_found(_error):
  return jsonify({"error": "API route not found."}), 404


@app.errorhandler(500)
def internal_error(_error):
  return jsonify({"error": "Internal server error."}), 500


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  app.run(host="0.0.0.0", port=port)
