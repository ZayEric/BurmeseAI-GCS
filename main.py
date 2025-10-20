from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os, logging
from threading import Lock

load_lock = Lock()
qa_pipe = None
qa_model = None
qa_tokenizer = None

WORKSPACE = "/workspace"
QA_BUCKET = "qa-model-bucket"
QA_PREFIX = "model/finetuned-qa-burmese"

def download_from_gcs(bucket_name, prefix, local_dir):
    """Download all files under prefix from GCS to local_dir, preserving folder structure."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logging.warning(f"No files found in gs://{bucket_name}/{prefix}")
        return

    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        # Skip empty blobs
        if blob.size == 0:
            continue
        # Remove the prefix from blob name
        rel_path = os.path.relpath(blob.name, prefix)
        # Flatten checkpoint folder if you want
        if rel_path.startswith("checkpoint/"):
            continue
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logging.info(f"Downloaded {blob.name} → {local_path}")

    logging.info(f"✅ Finished downloading {len(blobs)} files from {bucket_name}/{prefix}")

def get_qa_pipeline():
    """Lazy-load QA model safely for large safetensors."""
    global qa_pipe, qa_model, qa_tokenizer
    with load_lock:
        if qa_pipe is None:
            local_path = os.path.join(WORKSPACE, "qa")
            if not os.path.exists(local_path) or not os.listdir(local_path):
                logging.info("Downloading QA model from GCS...")
                download_from_gcs(QA_BUCKET, QA_PREFIX, local_path)

            logging.info(f"Loading QA tokenizer and model from {local_path}...")
            # Load tokenizer and model safely
            qa_tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
            qa_model = AutoModelForSeq2SeqLM.from_pretrained(local_path, device_map="auto")
            qa_pipe = pipeline("text2text-generation", model=qa_model, tokenizer=qa_tokenizer)
            logging.info("✅ QA model loaded")

    return qa_pipe
