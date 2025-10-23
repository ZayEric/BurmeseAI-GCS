import os
import logging
from threading import Lock
from flask import Flask, request, jsonify
import tempfile
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
import base64
import io
import soundfile as sf
from pydub import AudioSegment

device = 0 if torch.cuda.is_available() else -1
print(f"🔥 Using device: {'GPU' if device == 0 else 'CPU'}")


# ========== CONFIG ==========
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

WORKSPACE = "/workspace"
ASR_BUCKET = "speechtotext-model-bucket"
ASR_PREFIX = "model/finetuned-seamlessm4t-burmese"
QA_BUCKET = "qa-model-bucket"
QA_PREFIX = "model/finetuned-qa-burmese"

# ========== THREAD SAFETY ==========
asr_lock = Lock()
qa_lock = Lock()

# ========== GLOBAL MODELS ==========
asr_pipe = None
qa_pipe = None
asr_model = None
asr_processor = None
qa_model = None
qa_tokenizer = None


# ========== UTIL: GCS Download ==========
def download_from_gcs(bucket_name, prefix, local_dir):
    """Download all files recursively from GCS prefix to local_dir."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logging.warning(f"⚠️ No files found in gs://{bucket_name}/{prefix}")
        return

    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        if blob.size == 0:
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        if rel_path.startswith("checkpoint/"):
            continue
        dest_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
        logging.info(f"📦 {blob.name} → {dest_path}")

    logging.info(f"✅ Finished downloading {len(blobs)} files from gs://{bucket_name}/{prefix}")


# ========== ASR LOADER ==========
def get_asr_pipeline():
    global asr_pipe, asr_model, asr_processor
    with asr_lock:
        if asr_pipe is None:
            local_path = os.path.join(WORKSPACE, "asr")
            if not os.path.exists(local_path) or not os.listdir(local_path):
                logging.info("⬇️ Downloading ASR model from GCS...")
                download_from_gcs(ASR_BUCKET, ASR_PREFIX, local_path)

            logging.info(f"🚀 Loading ASR model from {local_path}...")
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            asr_processor = AutoProcessor.from_pretrained(local_path)
            asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(local_path, torch_dtype=dtype)
            asr_pipe = pipeline(
                "automatic-speech-recognition",
                model=asr_model,
                tokenizer=asr_processor.tokenizer,
                feature_extractor=asr_processor.feature_extractor,
                torch_dtype=dtype,
                device_map="auto"
            )
            logging.info("✅ ASR model loaded successfully.")
    return asr_pipe


# ========== QA LOADER ==========
def get_qa_pipeline():
    global qa_pipe, qa_model, qa_tokenizer
    with qa_lock:
        if qa_pipe is None:
            local_path = os.path.join(WORKSPACE, "qa")
            if not os.path.exists(local_path) or not os.listdir(local_path):
                logging.info("⬇️ Downloading QA model from GCS...")
                download_from_gcs(QA_BUCKET, QA_PREFIX, local_path)

            logging.info(f"🚀 Loading QA model from {local_path}...")
            qa_tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
            qa_model = AutoModelForSeq2SeqLM.from_pretrained(local_path, device_map="auto")
            qa_pipe = pipeline("text2text-generation", model=qa_model, tokenizer=qa_tokenizer)
            logging.info("✅ QA model loaded successfully.")
    return qa_pipe


# ========== HEALTH CHECK ==========
@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


# ========== ASR API ==========
@app.route("/asr", methods=["POST"])
def transcribe_audio():
    try:
        data = request.get_json()
        audio_base64 = data.get("audio_base64")
        if not audio_base64:
            return jsonify({"error": "Missing audio_base64"}), 400

        # Support URL or base64
        if audio_base64.startswith("http"):
            audio_bytes = requests.get(audio_base64).content
        else:
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


# ========== SPEECH → QA ==========
@app.route("/speechqa", methods=["POST"])
def speech_to_qa():
    try:
        data = request.get_json()
        audio_url = data.get("audio_base64")
        question = data.get("question", "")

        if not audio_url:
            return jsonify({"error": "Missing audio_base64"}), 400

        # Download or decode audio
        if audio_url.startswith("http"):
            audio_bytes = requests.get(audio_url).content
        else:
            audio_bytes = base64.b64decode(audio_url)

        # Convert to WAV
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
            try:
                AudioSegment.from_file(tmp_in_path).export(tmp_out_path, format="wav")
            except Exception:
                tmp_out_path = tmp_in_path

        # Run ASR
        asr = get_asr_pipeline()
        asr_result = asr(tmp_out_path, generate_kwargs={"tgt_lang": "mya"})
        transcript = asr_result[0]["text"] if isinstance(asr_result, list) else asr_result["text"]

        # Run QA
        qa = get_qa_pipeline()
        qa_result = qa(f"question: {question} context: {transcript}")
        answer = qa_result[0]["generated_text"]

        return jsonify({"transcript": transcript, "answer": answer})

    except Exception as e:
        logging.exception("SpeechQA error")
        return jsonify({"error": str(e)}), 500


# ========== TEXT QA ==========
@app.route("/textqa", methods=["POST"])
def text_to_qa():
    try:
        data = request.get_json()
        question = data.get("text", "")
        qa = get_qa_pipeline()
        qa_result = qa(f"question: {question}\nAnswer:")
        return jsonify({"answer": qa_result[0]["generated_text"]})
    except Exception as e:
        logging.exception("TextQA error")
        return jsonify({"error": str(e)}), 500


# ========== ENTRY ==========
if __name__ == "__main__":
    if torch.cuda.is_available():
        logging.info(f"🔥 GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("⚠️ GPU not available — using CPU")

    logging.info("Starting Flask app on port 8080...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
