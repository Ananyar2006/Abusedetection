import os
import requests
from flask import Flask, request, jsonify, send_from_directory
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
HF_MODEL = "joeddav/xlm-roberta-large-xnli"

SARVAM_API_KEY       = os.environ.get("SARVAM_API_KEY", "")
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"

LABELS = ["Non-Offensive", "Offensive"]

# ── Language codes ─────────────────────────────────────────────────────────────
SARVAM_LANG_CODES = {
    "hindi":     "hi-IN",
    "tamil":     "ta-IN",
    "kannada":   "kn-IN",
    "malayalam": "ml-IN",
}
GOOGLE_LANG_CODES = {
    "hindi":     "hi",
    "tamil":     "ta",
    "kannada":   "kn",
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
def predict(text: str, lang: str) -> dict:
    if not HF_API_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN is missing. Set it in Render Environment."
        )

    url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": LABELS,
            "hypothesis_template": "This example is {}.",
        },
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=45,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Hugging Face timed out. Please try again."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Hugging Face. "
            "Check the API endpoint and try again."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Hugging Face request failed."
        ) from exc

    if not response.ok:
        print("Hugging Face error:", response.status_code, response.text[:500])

        if response.status_code == 401:
            message = "Hugging Face token is invalid or unauthorized."
        elif response.status_code == 403:
            message = "Your token lacks the required inference permission."
        elif response.status_code == 429:
            message = "Hugging Face rate limit or usage quota reached."
        elif response.status_code in (404, 410):
            message = (
                "The model or inference route is unavailable. "
                "Check model availability on Hugging Face."
            )
        else:
            message = f"Hugging Face returned HTTP {response.status_code}."

        raise RuntimeError(message)

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Hugging Face returned an unexpected response."
        ) from exc

    # Validate the zero-shot classification response.
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unexpected model response: {str(data)[:300]}"
        )

    labels = data.get("labels", [])
    scores = data.get("scores", [])

    if not labels or not scores or len(labels) != len(scores):
        raise RuntimeError(
            "The model did not return valid classification scores."
        )

    label_scores = {
        str(label).lower(): float(score)
        for label, score in zip(labels, scores)
    }

    offensive_score = label_scores.get("offensive", 0.0)
    non_offensive_score = label_scores.get("non-offensive", 0.0)

    label = (
        "Offensive"
        if offensive_score > non_offensive_score
        else "Non-Offensive"
    )

    return {
        "label": label,
        "label_id": 1 if label == "Offensive" else 0,
        "confidence": round(max(offensive_score, non_offensive_score) * 100, 2),
        "language": lang,
        "text": text,
        "probs": {
            "non_offensive": round(non_offensive_score * 100, 2),
            "offensive": round(offensive_score * 100, 2),
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
