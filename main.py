import os
import logging
from threading import Lock
from flask import Flask, request, jsonify
from google.cloud import storage
from transformers import (
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import torch
import requests

# ---------- Configuration ----------
app = Flask(__name__)

WORKSPACE = "/workspace"

ASR_BUCKET = "speechtotext-model-bucket"
ASR_PREFIX = "model/finetuned-seamlessm4t-burmese"

QA_BUCKET = "qa-model-bucket"
QA_PREFIX = "model/finetuned-qa-burmese"

logging.basicConfig(level=logging.INFO)

# ---------- Lazy Loading Locks ----------
asr_lock = Lock()
qa_lock = Lock()

# ---------- Global Models ----------
asr_pipe = None
qa_pipe = None
asr_model = None
asr_processor = None
qa_model = None
qa_tokenizer = None

# ---------- Utility: GCS Download ----------
def download_from_gcs(bucket_name, prefix, local_dir):
    """Download files recursively from GCS preserving structure."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logging.warning(f"No files found in gs://{bucket_name}/{prefix}")
        return

    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        if blob.size == 0:
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        if rel_path.startswith("checkpoint/"):
            continue
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logging.info(f"Downloaded {blob.name} → {local_path}")

    logging.info(f"✅ Finished downloading {len(blobs)} files from {bucket_name}/{prefix}")

# ---------- Lazy Loading: ASR ----------
def get_asr_pipeline():
    global asr_pipe, asr_model, asr_processor
    with asr_lock:
        if asr_pipe is None:
            local_path = os.path.join(WORKSPACE, "asr")
            if not os.path.exists(local_path) or not os.listdir(local_path):
                logging.info("Downloading ASR model from GCS...")
                download_from_gcs(ASR_BUCKET, ASR_PREFIX, local_path)

            logging.info(f"Loading ASR model from {local_path}...")
            asr_processor = AutoProcessor.from_pretrained(local_path)
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(local_path, torch_dtype=torch.float16)
            asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=asr_model,
                tokenizer=asr_processor.tokenizer,
                feature_extractor=asr_processor.feature_extractor,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logging.info("✅ ASR model loaded")
    return asr_pipe

# ---------- Lazy Loading: QA ----------
def get_qa_pipeline():
    global qa_pipe, qa_model, qa_tokenizer
    with qa_lock:
        if qa_pipe is None:
            local_path = os.path.join(WORKSPACE, "qa")
            if not os.path.exists(local_path) or not os.listdir(local_path):
                logging.info("Downloading QA model from GCS...")
                download_from_gcs(QA_BUCKET, QA_PREFIX, local_path)

            logging.info(f"Loading QA model from {local_path}...")
            qa_tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
            qa_model = AutoModelForSeq2SeqLM.from_pretrained(local_path, device_map="auto")
            qa_pipe = pipeline("text2text-generation", model=qa_model, tokenizer=qa_tokenizer)
            logging.info("✅ QA model loaded")
    return qa_pipe

# ---------- API Routes ----------
@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/asr", methods=["POST"])
def transcribe_audio():
    """Receive base64 audio and return transcription."""
    try:
        data = request.get_json()
        audio_base64 = data.get("audio_base64")
        import base64
        import io
        import soundfile as sf

        if not audio_base64:
            return jsonify({"error": "Missing audio_base64"}), 400

        #audio_bytes = base64.b64decode(audio_base64)
        if audio_base64.startswith("http"):
            response = requests.get(audio_base64)
            audio_bytes = response.content
        else:
            # fallback if it’s real base64
            missing_padding = len(audio_base64) % 4
            if missing_padding:
                audio_base64 += '=' * (4 - missing_padding)
        audio_bytes = base64.b64decode(audio_base64)
        audio_stream = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_stream)

        asr = get_asr_pipeline()
        result = asr(audio, generate_kwargs={"max_new_tokens": 128})
        return jsonify({"text": result["text"]})

    except Exception as e:
        logging.exception("ASR error")
        return jsonify({"error": str(e)}), 500

# ---------- /speechqa ----------
@app.route("/speechqa", methods=["POST"])
def speech_to_qa():
    """Receive audio, transcribe, then answer a question."""
    try:
        data = request.get_json()
        audio_base64 = data.get("audio_base64")
        question = data.get("question", "")

        if not audio_base64:
            return jsonify({"error": "Missing audio_base64"}), 400

        import base64, io, soundfile as sf

        # Decode audio

        audio_url = audio_base64
        if not audio_url:
            return JSONResponse({"error": "Missing audio_url"}, status_code=400)

        logging.info(f"Url starts with: {audio_url[:100]}")

        if audio_url.startswith("http"):
            resp = requests.get(audio_url)
            audio_bytes = resp.content
        else:
            audio_bytes = base64.b64decode(audio_url)
            
        audio_stream = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_stream)

        # Run ASR
        asr = get_asr_pipeline()
        asr_result = asr(audio, generate_kwargs={"max_new_tokens": 128})
        transcript = asr_result["text"]

        # Run QA
        qa = get_qa_pipeline()
        qa_result = qa(f"question: {question} context: {transcript}")
        #qa_result = qa(f"question: {question} context: {question}")
        answer = qa_result[0]["generated_text"]

        return jsonify({
            "transcript": transcript,
            "answer": answer
        })
    except Exception as e:
        logging.exception("SpeechQA error")
        return jsonify({"error": str(e)}), 500

# ---------- /textqa ----------
@app.route("/textqa", methods=["POST"])
def text_to_qa():
    """Receive question in text, then answer a question."""
    try:
        data = request.get_json()
        question = data.get("text", "")

        # Run QA
        qa = get_qa_pipeline()
        qa_result = qa(f"question: {question}\nAnswer:")
        answer = qa_result[0]["generated_text"]

        return jsonify({
            "answer": answer
        })
    except Exception as e:
        logging.exception("TextQA error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.info("Preloading QA model at startup...")
    get_qa_pipeline()  # Force model download once
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
