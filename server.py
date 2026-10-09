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

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": text,
        "parameters": {
            "candidate_labels": LABELS,
            "hypothesis_template": "This example is {}."
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    labels = data.get("labels", [])
    scores = data.get("scores", [])

    label = labels[0] if labels else "Non-Offensive"
    confidence = round(scores[0] * 100, 2) if scores else 0

    probs = {"non_offensive": 0, "offensive": 0}
    for i, lbl in enumerate(labels):
        val = round(scores[i] * 100, 2)
        if lbl == "Non-Offensive":
            probs["non_offensive"] = val
        elif lbl == "Offensive":
            probs["offensive"] = val

    return {
        "label": label,
        "label_id": 1 if label == "Offensive" else 0,
        "confidence": confidence,
        "language": lang,
        "text": text,
        "probs": probs
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
