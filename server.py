
import os
import logging
import requests

from flask import Flask, request, jsonify, send_from_directory
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
# Set these in Render Environment Variables.
# Never put secret keys in frontend JavaScript.
# --------------------------------------------------

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "").strip()
HF_MODEL = "facebook/bart-large-mnli"

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "").strip()
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

LABELS = ["Non-Offensive", "Offensive"]

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

    # Try Sarvam first.
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

            app.logger.warning(
                "Sarvam translation returned HTTP %s",
                response.status_code,
            )

        except (requests.RequestException, ValueError) as exc:
            app.logger.warning("Sarvam translation failed: %s", exc)

    # Optional fallback.
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
# HUGGING FACE CLASSIFICATION
# --------------------------------------------------

def predict(text, lang):
    if not HF_API_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN is missing. Add it to Render Environment."
        )

    url = (
        "https://router.huggingface.co/"
        f"hf-inference/models/{HF_MODEL}"
    )

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": LABELS,
            "hypothesis_template": "This example is {}.",
        },
    }

    try:
        response = session.post(
            url,
            headers={
                "Authorization": f"Bearer {HF_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Hugging Face timed out. Please try again."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Hugging Face. "
            "Check the Render service network connection."
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "The Hugging Face request failed."
        ) from exc

    if not response.ok:
        app.logger.error(
            "Hugging Face HTTP %s: %s",
            response.status_code,
            response.text[:500],
        )

        if response.status_code == 401:
            message = "Hugging Face authentication failed."
        elif response.status_code == 403:
            message = "Your Hugging Face token lacks permission."
        elif response.status_code == 429:
            message = "Hugging Face rate limit or quota reached."
        elif response.status_code in (404, 410):
            message = (
                "The model or inference route is unavailable. "
                "Check the model on Hugging Face."
            )
        else:
            message = (
                f"Hugging Face returned HTTP {response.status_code}."
            )

        raise RuntimeError(message)

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Hugging Face returned invalid JSON."
        ) from exc

    # Some APIs return an error object or another response shape.
    if not isinstance(data, dict):
        raise RuntimeError(
            "Unexpected classification response from Hugging Face."
        )

    labels = data.get("labels")
    scores = data.get("scores")

    if (
        not isinstance(labels, list)
        or not isinstance(scores, list)
        or len(labels) != len(scores)
        or not labels
    ):
        raise RuntimeError(
            "No valid classification scores were returned."
        )

    score_map = {
        str(label).strip().lower(): float(score)
        for label, score in zip(labels, scores)
    }

    offensive_score = score_map.get("offensive", 0.0)
    non_offensive_score = score_map.get("non-offensive", 0.0)

    label = (
        "Offensive"
        if offensive_score > non_offensive_score
        else "Non-Offensive"
    )

    confidence = max(
        offensive_score,
        non_offensive_score,
    ) * 100

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
        "translator_fallback_available": TRANSLATOR_AVAILABLE,
    })


# --------------------------------------------------
# ANALYZE TEXT
# --------------------------------------------------

@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a valid JSON request."
        }), 400

    text = data.get("text", "")
    lang = str(data.get("language", "hindi")).lower().strip()

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please enter some text."}), 400

    text = text.strip()

    if len(text) > 5000:
        return jsonify({
            "error": "Text must be 5000 characters or fewer."
        }), 400

    if lang not in SARVAM_LANG_CODES:
        return jsonify({
            "error": "Unsupported language. Choose Hindi, Tamil, Kannada or Malayalam."
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
        return jsonify({
            "error": "An unexpected server error occurred."
        }), 500


# --------------------------------------------------
# ANALYZE A PROVIDED SPEECH TRANSCRIPT
# This route does NOT transcribe uploaded audio.
# --------------------------------------------------

@app.route("/analyze-speech", methods=["POST"])
def analyze_speech():
    transcript = request.form.get("transcript", "").strip()
    lang = request.form.get("language", "hindi").lower().strip()

    if not transcript:
        return jsonify({
            "error": "No speech transcript was provided."
        }), 400

    if lang not in SARVAM_LANG_CODES:
        return jsonify({"error": "Unsupported language."}), 400

    try:
        result = predict(transcript, lang)
        result["transcript"] = transcript
        result["translation"] = translate_to_english(
            transcript, lang
        )

        return jsonify(result)

    except RuntimeError as exc:
        app.logger.error("Speech analysis failed: %s", exc)
        return jsonify({"error": str(exc)}), 502

    except Exception:
        app.logger.exception("Unexpected speech analysis error")
        return jsonify({
            "error": "An unexpected server error occurred."
        }), 500


# --------------------------------------------------
# ERROR HANDLERS
# --------------------------------------------------

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "error": "The request is too large."
    }), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({
        "error": "API route not found."
    }), 404


@app.errorhandler(500)
def internal_error(_error):
    return jsonify({
        "error": "Internal server error."
    }), 500


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
