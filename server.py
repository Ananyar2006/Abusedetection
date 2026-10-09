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
# API CONFIGURATION
# --------------------------------------------------

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "").strip()
HF_MODEL = "xlm-roberta-base"
LABELS = ["Non-Offensive", "Offensive"]

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
          source=GOOGLE_LANG_CODES[lang], target="en"
      ).translate(text)
    except Exception as exc:
      app.logger.warning("Google translation failed: %s", exc)

  return ""


# --------------------------------------------------
# REMOTE HUGGING FACE CLASSIFICATION API
# --------------------------------------------------


# --------------------------------------------------
# REMOTE HUGGING FACE CLASSIFICATION API
# --------------------------------------------------


def predict(text, lang):
  if not HF_API_TOKEN:
    raise RuntimeError(
        "HF_API_TOKEN is missing. Add it to Render Environment Variables."
    )

  url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

  # FIXED: xlm-roberta-base is standard classification, so send ONLY inputs.
  payload = {
      "inputs": text,
  }

  try:
    response = session.post(
        url,
        headers={
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
  except Exception as exc:
    raise RuntimeError("Could not connect to Hugging Face API.") from exc

  if not response.ok:
    raise RuntimeError(f"Hugging Face returned HTTP {response.status_code}.")

  data = response.json()

  # Handle standard text-classification list responses (e.g., [[{'label': 'LABEL_1', 'score': 0.9}, ...]])
  if isinstance(data, list) and len(data) > 0:
    if isinstance(data[0], list):
      scores_list = data[0]
    else:
      scores_list = data
  else:
    scores_list = []

  score_map = {
      str(item.get("label", "")).strip().lower(): float(item.get("score", 0.0))
      for item in scores_list
  }

  # Map standard model labels (e.g., offensive/non-offensive or LABEL_1/LABEL_0)
  offensive_score = score_map.get(
      "offensive", score_map.get("label_1", score_map.get("1", 0.0))
  )
  non_offensive_score = score_map.get(
      "non-offensive",
      score_map.get(
          "non_offensive", score_map.get("label_0", score_map.get("0", 0.0))
      ),
  )

  # Fallback if labels are generic
  if (
      offensive_score == 0.0
      and non_offensive_score == 0.0
      and len(scores_list) >= 2
  ):
    offensive_score = scores_list[1].get("score", 0.0)
    non_offensive_score = scores_list[0].get("score", 0.0)

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
      "hf_token_set": bool(HF_API_TOKEN),
      "sarvam_key_set": bool(SARVAM_API_KEY),
  })


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
  data = request.get_json(silent=True)
  if not isinstance(data, dict):
    return jsonify({"error": "Send a valid JSON request."}), 400

  text = data.get("text", "").strip()
  lang = str(data.get("language", "hindi")).lower().strip()

  if not text:
    return jsonify({"error": "Please enter some text."}), 400

  if lang not in SARVAM_LANG_CODES:
    return jsonify({"error": "Unsupported language."}), 400

  try:
    result = predict(text, lang)
    result["translation"] = translate_to_english(text, lang)
    return jsonify(result)
  except RuntimeError as exc:
    return jsonify({"error": str(exc)}), 502
  except Exception:
    return jsonify({"error": "An unexpected server error occurred."}), 500


@app.route("/analyze-speech", methods=["POST"])
def analyze_speech():
  transcript = request.form.get("transcript", "").strip()
  lang = request.form.get("language", "hindi").lower().strip()

  if not transcript:
    return jsonify({"error": "No speech transcript was provided."}), 400

  try:
    result = predict(transcript, lang)
    result["transcript"] = transcript
    result["translation"] = translate_to_english(transcript, lang)
    return jsonify(result)
  except RuntimeError as exc:
    return jsonify({"error": str(exc)}), 502
  except Exception:
    return jsonify({"error": "An unexpected server error occurred."}), 500


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  app.run(host="0.0.0.0", port=port)
