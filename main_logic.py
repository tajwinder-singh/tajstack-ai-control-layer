import fcntl
import hashlib
import time
import shutil
import calendar
import threading
from threading import Lock
import tempfile
from flask import jsonify
import psycopg2
from psycopg2.errors import SerializationFailure
from psycopg2.extras import Json
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extensions import ISOLATION_LEVEL_SERIALIZABLE
from bs4 import BeautifulSoup
import os
import joblib
import re
import json
import logging
import uuid
import pdfplumber
from datetime import datetime, timezone, timedelta
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import pipeline
from dotenv import load_dotenv
import requests
import ipaddress
import socket
from urllib.parse import urlparse

load_dotenv()

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

logger = logging.getLogger("rag_system")


# -----------------------
# API ERROR HELPER
# -----------------------
def api_error(message: str, status_code: int = 400, extra: dict | None = None):
    payload = {"error": message}
    if extra:
        payload.update(extra)
    return jsonify(payload), status_code


# -----------------------
# SAFE LOGGING
# -----------------------
# Wrapped to prevent logging crashes from killing request handling.
# print() is intentionally used inside the except blocks here —
# if the logger itself fails, we still need some stdout signal.
def safe_log_error(message: str, *args):
    try:
        logger.error(message, *args, exc_info=True)
    except Exception:
        print(f"Logger failed. Error: {message}")

def safe_log_info(message: str, *args):
    try:
        logger.info(message, *args)
    except Exception:
        print(f"Logger failed. Info: {message}")

def safe_log_warning(message: str, *args):
    try:
        logger.warning(message, *args)
    except Exception:
        print(f"Logger failed. Warning: {message}")


if faiss is None:
    safe_log_error("FAISS not installed. System cannot start.")
    raise RuntimeError("Critical dependency missing: faiss")
if np is None:
    safe_log_error("Numpy not installed. System cannot start.")
    raise RuntimeError("Critical dependency missing: numpy")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    PG_DSN = os.environ["PG_DSN"]
except Exception:
    safe_log_error("PG_DSN failed to load.")
    raise RuntimeError("Missing env var: PG_DSN")

if not PG_DSN:
    raise ValueError("PG_DSN is empty")


# -----------------------
# CONSTANTS
# -----------------------
MAX_VECTORS = 20000
MAX_DISK_BYTES = 200 * 1024 * 1024     # 200MB
MAX_KB_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB per file
MIN_ALLOWED_KB_TEXT_CHARS = 100

ALLOWED_KB_EXTENSIONS = {".txt", ".pdf", ".json"}

ESTIMATED_COST_PER_LLM_CALL = 5  # INR

AUTO_BLOCK_MAX_LLM_CALLS_PER_MONTH = 100000
AUTO_BLOCK_MAX_REQUESTS_PER_MONTH = 300000
EXPECTED_MONTHLY_LLM_CALLS = 10000

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

USAGE_ROOT = os.path.join(BASE_DIR, "usage", "monthly")
os.makedirs(USAGE_ROOT, exist_ok=True)

SECURITY_CAP_ROOT = os.path.join(BASE_DIR, "security", "caps")
os.makedirs(SECURITY_CAP_ROOT, exist_ok=True)

INTERACTION_TTL = timedelta(hours=2)


# -----------------------
# DIRECTORY STRUCTURE
# -----------------------
"""
knowledge_base/     ← permanent KB files (placed manually)
uploads/            ← KB files ingested via /api/ingest-kb
vector_store/
    kb/             ← FAISS index + metadata for knowledge_base/
    uploads/        ← FAISS index + metadata for uploads/
"""
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
VECTOR_KB_DIR = os.path.join(BASE_DIR, "vector_store", "kb")
VECTOR_UPLOADS_DIR = os.path.join(BASE_DIR, "vector_store", "uploads")

for d in [KB_DIR, UPLOADS_DIR, VECTOR_KB_DIR, VECTOR_UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)

KB_INDEX_PATH = os.path.join(VECTOR_KB_DIR, "index.faiss")
KB_META_PATH = os.path.join(VECTOR_KB_DIR, "meta.json")
UP_INDEX_PATH = os.path.join(VECTOR_UPLOADS_DIR, "index.faiss")
UP_META_PATH = os.path.join(VECTOR_UPLOADS_DIR, "meta.json")
KB_LOCK_PATH = os.path.join(BASE_DIR, ".kb_ingest.lock")


# -----------------------
# BERT MODEL (LAZY-LOADED)
# -----------------------
# Lazy-loading with double-checked locking ensures the model is loaded only once per worker process.
# The lock prevents duplicate initialization if multiple threads try to load it at the same time.
_embedding_model = None
_model_lock = threading.Lock()

def load_bert_model():
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu"
                )
    return _embedding_model


# -----------------------
# INTENT MODEL (LAZY-LOADED)
# -----------------------
INTENT_MODEL_PATH = os.path.join(BASE_DIR, "intent_model_embedder")
CLF_PATH = os.path.join(BASE_DIR, "intent_classifier.joblib")

embedder = None
clf = None
intent_model_loaded = False
intent_model_lock = Lock()

id2label = {
    0: "refund_request",
    1: "order_status",
    2: "technical_issue",
    3: "complaint",
    4: "general_query",
}


def load_intent_model():
    global embedder, clf

    if embedder is None or clf is None:
        with intent_model_lock:
            if embedder is None:
                try:
                    embedder = SentenceTransformer(INTENT_MODEL_PATH)
                except Exception:
                    raise
            if clf is None:
                try:
                    clf = joblib.load(CLF_PATH)
                except Exception:
                    raise


try:
    load_intent_model()
    intent_model_loaded = True
    safe_log_info("Intent model and classifier loaded successfully.")
except Exception:
    safe_log_error("Intent model failed to load.")
    intent_model_loaded = False


if not intent_model_loaded:
    def predict_intent(message: str):
        return {
            "intent": "general_query",
            "intent_confidence": 0.0,
            "intent_model_used": False,
            "intent_status": "Intent model not loaded.",
        }
else:
    def predict_intent(message: str):
        load_intent_model()

        if not message:
            return {
                "intent": "general_query",
                "intent_confidence": 0.0,
                "intent_model_used": False,
                "intent_status": "Empty input.",
            }

        try:
            embedding = embedder.encode([message])
            probs = clf.predict_proba(embedding)[0]
            pred = np.argmax(probs)
            confidence = probs[pred]

            return {
                "intent": id2label.get(pred, "general_query"),
                "intent_confidence": float(confidence),
                "intent_model_used": True,
                "intent_status": "Intent successfully predicted.",
            }

        except Exception as e:
            safe_log_error("Intent prediction failed.")
            return {
                "intent": "general_query",
                "intent_confidence": 0.0,
                "intent_model_used": False,
                "intent_status": f"Exception: {str(e)}",
            }


# -----------------------
# SENTIMENT MODEL (LAZY-LOADED)
# -----------------------
_sentiment_pipeline = None
_sentiment_lock = threading.Lock()

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        with _sentiment_lock:
            if _sentiment_pipeline is None:
                try:
                    _sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1,
                    )
                except Exception:
                    safe_log_error("Sentiment model failed to load. Continuing without it.")
                    _sentiment_pipeline = False
    return _sentiment_pipeline


def predict_sentiment(customer_context: str):
    """Returns (label, score) or ("", 0.0) if unavailable."""
    if not customer_context or not customer_context.strip():
        return "", 0.0

    try:
        pipeline_fn = get_sentiment_pipeline()
        if not pipeline_fn:
            return "", 0.0

        result = pipeline_fn(customer_context[:1000])
        if isinstance(result, list) and len(result) > 0:
            return result[0]["label"], float(result[0]["score"])

    except Exception:
        safe_log_error("Sentiment prediction failed.")

    return "", 0.0


def adapt_tone_for_sentiment(original_tone: str, sentiment_label: str,
                              sentiment_score: float, intent: str):
    if not sentiment_label:
        return original_tone

    if sentiment_label.upper() == "NEGATIVE":
        if intent in ("complaint", "technical_issue", "refund_request"):
            return "empathetic"
        return "professional"

    return original_tone


# -----------------------
# FAISS INDEX HELPERS
# -----------------------
def _fsync_parent_dir(path: str) -> None:
    """
    Syncs the parent directory after an atomic file replace.
    Without this, a crash immediately after os.replace() can leave
    the directory entry pointing to the old file on some filesystems.
    """
    parent = os.path.dirname(path) or "."
    fd = os.open(parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_faiss_index(index, vector_path, meta_path, metadata_list):
    """
    Saves FAISS index and metadata atomically.

    We write to temporary files first, then replace the old files only after
    the new ones are fully written. This helps avoid partial writes or
    corrupted indexes if the process crashes during a save.
    fcntl file locking prevents concurrent workers from writing at the same time.
    """
    if index is None or not isinstance(metadata_list, list):
        safe_log_error("Invalid inputs to save_faiss_index.")
        return

    if hasattr(index, "ntotal") and index.ntotal != len(metadata_list):
        safe_log_error(
            "FAISS/metadata mismatch before save: index.ntotal=%s, metadata_count=%s",
            index.ntotal, len(metadata_list)
        )

    index_tmp = vector_path + ".tmp"
    meta_tmp = meta_path + ".tmp"

    with open(KB_LOCK_PATH, "a+") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        try:
            faiss.write_index(index, index_tmp)

            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(index_tmp, vector_path)
            _fsync_parent_dir(vector_path)

            os.replace(meta_tmp, meta_path)
            _fsync_parent_dir(meta_path)

        except Exception:
            for p in (index_tmp, meta_tmp):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            safe_log_error("Failed to save FAISS index.")


def load_faiss_index(vector_path, meta_path):
    """
    Loads FAISS index with the same file lock used during writes.
    We lock on reads too — if a write is happening on another worker
    and we read mid-write, we'd get corrupted metadata.
    """
    with open(KB_LOCK_PATH, "a+") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        if not os.path.exists(vector_path) or not os.path.exists(meta_path):
            return None, []

        try:
            index = faiss.read_index(vector_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)
        except Exception:
            safe_log_error("Failed to load FAISS index or metadata.")
            return None, []

        if not isinstance(metadata_list, list):
            safe_log_error("Invalid metadata format (expected list).")
            return None, []

        if index.ntotal != len(metadata_list):
            safe_log_error(
                "FAISS mismatch: index.ntotal=%s, metadata_count=%s",
                index.ntotal, len(metadata_list)
            )
            return None, []

        return index, metadata_list


# -----------------------
# FILE HELPERS
# -----------------------
def _today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _current_month():
    return datetime.now(timezone.utc).strftime("%Y-%m")

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Prevents path traversal and unsafe characters.
    Uses underscore replacement (not removal) to preserve readability.
    """
    filename = os.path.basename(filename)
    filename = filename.replace("..", "")
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    filename = re.sub(r"_+", "_", filename)
    return filename.lower()


def create_secure_temp_dir(request_id: str) -> str:
    """
    mode=0o700: only the process owner can read/write/execute.
    Prevents other OS processes from accessing the file while it's
    being validated — protects against symlink attacks and race conditions.
    """
    base = f"/tmp/uploads/{request_id}"
    os.makedirs(base, mode=0o700, exist_ok=True)
    return base


def cleanup_orphan_temp_dirs(base):
    """Called at app startup to remove temp dirs left by crashed workers."""
    if not os.path.exists(base):
        return

    for request_id in os.listdir(base):
        path = os.path.join(base, request_id)
        try:
            shutil.rmtree(path)
        except Exception:
            safe_log_error("Failed to remove orphan temp dir: %s", path)


def validate_kb_filename(filename: str):
    if not filename:
        return {
            "status": "rejected",
            "message": "Filename is required.",
            "message_for_client": "The uploaded file type is not supported.",
            "code": "MISSING_FILE_NAME",
            "status_code": 400,
        }

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_KB_EXTENSIONS:
        return {
            "status": "rejected",
            "message": f"Unsupported file type: {filename}",
            "message_for_client": "The uploaded file type is not supported.",
            "code": "UNSUPPORTED_FILE_TYPE",
            "status_code": 400,
        }

    return {"status": "success"}


def validate_file_size_for_temp(file_path: str):
    size = os.path.getsize(file_path)
    if size > MAX_KB_FILE_SIZE_BYTES:
        raise ValueError("File exceeds maximum allowed size (5MB)")


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path.lower())[1]

    if ext == ".txt":
        try:
            with open(file_path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace").strip()
        except Exception:
            safe_log_error("Text extraction failed for: %s", file_path)
            raise

    if ext == ".pdf":
        try:
            return extract_text_from_kb_pdf(file_path)
        except Exception:
            raise

    if ext == ".json":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            safe_log_error("JSON load failed for: %s", file_path)
            raise
        return convert_policy_json_to_text(data)

    raise ValueError(f"Unsupported file type: {ext}")


def extract_text_from_kb_pdf(file_path: str) -> str:
    text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception:
        safe_log_error("PDF extraction failed for: %s", file_path)
        raise
    return "\n".join(text)


def convert_policy_json_to_text(data: dict) -> str:
    """
    Converts policy.json into semantic plain text for embedding.
    JSON structure is not understood by the embedding model — converting
    to natural language phrasing significantly improves retrieval quality.
    """
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key.replace('_', ' ').title()}:")
            for item in value:
                lines.append(f"- {item}")
        else:
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
    return "\n".join(lines)


def sanitize_kb_text(text: str) -> str:
    """
    Removes boilerplate and noise before embedding.
    Boilerplate text (copyright, disclaimers, page numbers) produces
    noise vectors that degrade retrieval — removing it before indexing
    keeps the vector space clean and retrieval quality high.
    """
    if not text:
        raise ValueError("Empty KB content")

    text = text.replace("\x00", "").strip()
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    for pattern in [
        r"(?i)this document.*confidential.*",
        r"(?i)do not distribute.*",
        r"(?i)all rights reserved.*",
        r"(?i)terms and conditions.*",
        r"(?i)privacy policy.*",
        r"(?i)copyright.*",
        r"(?i)page \d+ of \d+",
        r"[-_=]{5,}",
    ]:
        text = re.sub(pattern, "", text)

    text = text.strip()

    if len(text) < MIN_ALLOWED_KB_TEXT_CHARS:
        raise ValueError("KB content too short to be meaningful")
    if not re.search(r"[a-zA-Z0-9]{3,}", text):
        raise ValueError("KB content lacks meaningful tokens")

    return text


def store_raw_kb_file(*, request_id: str, filename: str, raw_bytes: bytes) -> str:
    """
    Stores the raw uploaded file as-is in binary, read-only (0o400).
    This is the immutable audit copy — if a client later claims the file
    we ingested doesn't match what they sent, we can produce this record.
    """
    base_dir = os.path.join(BASE_DIR, "raw_files", request_id)
    os.makedirs(base_dir, mode=0o700, exist_ok=True)

    safe_name = sanitize_filename(filename)
    final_path = os.path.join(base_dir, safe_name)

    with open(final_path, "wb") as f:
        f.write(raw_bytes)

    os.chmod(final_path, 0o400)  # immutable after write
    return final_path


def store_cleaned_kb_text(*, safe_filename: str, cleaned_text: str,
                           overwrite: bool = False) -> str:
    """
    Stores cleaned KB text as .txt regardless of original extension.
    We force .txt because uploads/ and knowledge_base/ contain only
    cleaned text — the original format is preserved in raw_files/.
    Atomic write via NamedTemporaryFile + os.replace prevents partial
    writes from corrupting the file on crash.
    """
    name_without_ext = os.path.splitext(safe_filename)[0]
    final_filename = f"{name_without_ext}.txt"
    final_path = os.path.join(UPLOADS_DIR, final_filename)

    if overwrite and os.path.exists(final_path):
        os.chmod(final_path, 0o600)
        os.remove(final_path)
        clear_uploads_vector_store()

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, dir=UPLOADS_DIR
    ) as tmp:
        tmp.write(cleaned_text)
        tmp_path = tmp.name

    os.replace(tmp_path, final_path)
    os.chmod(final_path, 0o400)  # immutable after write
    return final_path


def get_all_kb_files():
    if not os.path.exists(UPLOADS_DIR):
        return []
    files = []
    for root, _, filenames in os.walk(UPLOADS_DIR):
        for name in filenames:
            files.append(os.path.join(root, name))
    return files


def clear_uploads_vector_store():
    """
    Clears FAISS index + metadata for uploads.
    Called before a re-index to prevent stale vectors from a failed
    partial upload accumulating in the vector store.
    """
    for p in (UP_INDEX_PATH, UP_META_PATH):
        if os.path.exists(p):
            os.remove(p)


def rebuild_index(bert_model):
    """Rebuilds the uploads vector store from whatever files exist in UPLOADS_DIR."""
    file_paths = get_all_kb_files()
    if not file_paths:
        return {"status": "success"}

    for path in file_paths:
        result = index_uploaded_file(path, bert_model)
        if result and result.get("status") == "error":
            return {"status": "error", "message": result["message"]}

    return {"status": "success"}


# -----------------------
# USAGE TRACKING (JSON + PostgreSQL)
# -----------------------
"""
Usage is tracked in both JSON (fast, local) and PostgreSQL (durable, queryable).
JSON is the low-latency write path. PostgreSQL is the source of truth.
Both are updated on every request — they should always agree.
"""

def load_monthly_usage():
    month = _current_month()
    path = os.path.join(USAGE_ROOT, f"{month}.json")

    f = open(path, "a+")
    fcntl.flock(f, fcntl.LOCK_EX)  # exclusive lock held until save_monthly_usage releases it

    f.seek(0)
    try:
        data = json.load(f)
    except Exception:
        data = {
            "month": month,
            "total_requests": 0,
            "llm_calls": 0,
            "estimated_cost": 0,
            "peak_day": None,
            "daily_llm_calls": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    if data.get("month") != month:
        data.update({
            "month": month,
            "total_requests": 0,
            "llm_calls": 0,
            "estimated_cost": 0,
            "peak_day": None,
            "daily_llm_calls": {},
        })

    return data, f


def save_monthly_usage(f, data: dict, update_timestamp: bool = False):
    if update_timestamp:
        data["last_updated_at"] = datetime.now(timezone.utc).isoformat()

    f.seek(0)
    f.truncate()
    json.dump(data, f)
    f.flush()
    os.fsync(f.fileno())

    fcntl.flock(f, fcntl.LOCK_UN)
    f.close()


def record_request_usage(*, tenant_id: str = "default"):
    """Records each API request in both JSON and PostgreSQL."""
    data, f = load_monthly_usage()
    try:
        data["total_requests"] += 1
    finally:
        save_monthly_usage(f, data, True)

    # PostgreSQL — serializable isolation prevents concurrent
    # increment races when multiple workers handle requests simultaneously
    for attempt in range(3):
        conn = get_pg_conn()
        try:
            conn.set_session(isolation_level=ISOLATION_LEVEL_SERIALIZABLE)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO monthly_usage (tenant_id, month, total_requests)
                    VALUES (%s, %s, 1)
                    ON CONFLICT (tenant_id, month)
                    DO UPDATE SET
                        total_requests = monthly_usage.total_requests + 1,
                        last_updated_at = NOW();
                    """,
                    (tenant_id, _current_month())
                )
            conn.commit()
            break
        except SerializationFailure:
            conn.rollback()
            if attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))
        finally:
            release_pg_conn(conn)


def record_llm_usage(*, tenant_id: str = "default"):
    """Tracks LLM calls with daily granularity for peak-day detection."""
    data, f = load_monthly_usage()
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        data["llm_calls"] += 1
        data["estimated_cost"] += ESTIMATED_COST_PER_LLM_CALL

        daily = data.setdefault("daily_llm_calls", {})
        daily[today] = daily.get(today, 0) + 1

        if data["peak_day"] is None or daily[today] > daily.get(data["peak_day"], 0):
            data["peak_day"] = today
    finally:
        save_monthly_usage(f, data, True)

    for attempt in range(3):
        conn = get_pg_conn()
        try:
            conn.set_session(isolation_level=ISOLATION_LEVEL_SERIALIZABLE)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO monthly_usage (tenant_id, month, llm_calls, estimated_ai_cost)
                    VALUES (%s, %s, 1, %s)
                    ON CONFLICT (tenant_id, month)
                    DO UPDATE SET
                        llm_calls = monthly_usage.llm_calls + 1,
                        estimated_ai_cost = monthly_usage.estimated_ai_cost + EXCLUDED.estimated_ai_cost,
                        last_updated_at = CURRENT_DATE;
                    """,
                    (tenant_id, _current_month(), ESTIMATED_COST_PER_LLM_CALL)
                )
                cur.execute(
                    """
                    INSERT INTO daily_llm_usage (tenant_id, usage_date, llm_calls)
                    VALUES (%s, CURRENT_DATE, 1)
                    ON CONFLICT (tenant_id, usage_date)
                    DO UPDATE SET llm_calls = daily_llm_usage.llm_calls + 1;
                    """,
                    (tenant_id,)
                )
                # Update peak day based on actual daily counts
                cur.execute(
                    """
                    UPDATE monthly_usage mu
                    SET peak_day = sub.usage_date
                    FROM (
                        SELECT usage_date
                        FROM daily_llm_usage
                        WHERE tenant_id = %s
                        AND usage_date >= date_trunc('month', CURRENT_DATE)
                        ORDER BY llm_calls DESC, usage_date ASC
                        LIMIT 1
                    ) sub
                    WHERE mu.tenant_id = %s AND mu.month = %s;
                    """,
                    (tenant_id, tenant_id, _current_month())
                )
            conn.commit()
            break
        except SerializationFailure:
            conn.rollback()
            if attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))
        finally:
            release_pg_conn(conn)


# -----------------------
# AUTO-BLOCK SECURITY
# -----------------------
"""
Per-API-key monthly caps on requests and LLM calls.
If a key exceeds the threshold — whether due to a bug, an infinite
loop, or genuine abuse — the key is auto-disabled so costs don't spiral.

security/caps/<api_key>.json
"""

def load_security_state(api_key: str):
    _ensure_dir(SECURITY_CAP_ROOT)
    path = os.path.join(SECURITY_CAP_ROOT, f"{api_key}.json")
    month = _current_month()

    f = open(path, "a+")
    fcntl.flock(f, fcntl.LOCK_EX)

    f.seek(0)
    try:
        data = json.load(f)
    except Exception:
        data = {
            "api_key": api_key,
            "month": month,
            "request_count": 0,
            "llm_call_count": 0,
            "disabled": False,
            "disabled_reason": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    if data.get("month") != month:
        data.update({
            "month": month,
            "request_count": 0,
            "llm_call_count": 0,
            "disabled": False,
            "disabled_reason": None,
        })

    return data, f


def save_security_state(f, data: dict, update_timestamp: bool):
    if update_timestamp:
        data["last_updated_at"] = datetime.now(timezone.utc).isoformat()

    f.seek(0)
    f.truncate()
    json.dump(data, f)
    f.flush()
    os.fsync(f.fileno())

    fcntl.flock(f, fcntl.LOCK_UN)
    f.close()


def enforce_auto_block_precheck(api_key: str):
    data, f = load_security_state(api_key)
    try:
        if data.get("disabled"):
            return False, data.get("disabled_reason") or "api_key_disabled"
        return True, None
    finally:
        save_security_state(f, data, False)


def record_security_requests(api_key: str):
    data, f = load_security_state(api_key)
    try:
        data["request_count"] += 1
        if data["request_count"] > AUTO_BLOCK_MAX_REQUESTS_PER_MONTH:
            data["disabled"] = True
            data["disabled_reason"] = "auto_block_requests_cap_exceeded"
        return not data["disabled"], data.get("disabled_reason")
    finally:
        save_security_state(f, data, True)


def record_security_llm_call(api_key: str):
    data, f = load_security_state(api_key)
    try:
        data["llm_call_count"] += 1
        if data["llm_call_count"] > AUTO_BLOCK_MAX_LLM_CALLS_PER_MONTH:
            data["disabled"] = True
            data["disabled_reason"] = "auto_block_llm_cap_exceeded"
        return not data["disabled"], data.get("disabled_reason")
    finally:
        save_security_state(f, data, True)


# -----------------------
# MONITORING ALERTS
# -----------------------
"""
Daily cron job checks these conditions and logs alerts.
Designed to catch: sudden traffic spikes, runaway automation,
approaching auto-block threshold, and sustained week-over-week growth.
"""

def alert_daily_spike(usage: dict):
    daily = usage.get("daily_llm_calls", {})
    if len(daily) < 2:
        return None
    dates = sorted(daily.keys())
    today, yesterday = dates[-1], dates[-2]
    if daily[today] > 3 * daily[yesterday]:
        return f"DAILY_SPIKE: {daily[today]} vs {daily[yesterday]}"
    return None


def alert_fast_monthly_burn(usage: dict):
    llm_calls = usage.get("llm_calls", 0)
    created = datetime.fromisoformat(usage["created_at"])
    year, month = map(int, usage["month"].split("-"))
    month_start = datetime(year, month, 1, tzinfo=timezone.utc)
    days_in_month = calendar.monthrange(year, month)[1]
    burn_start = max(created, month_start)
    now = datetime.now(timezone.utc)
    days_passed = (now - burn_start).days + 1
    days_remaining_window = (
        datetime(year, month, days_in_month, tzinfo=timezone.utc) - burn_start
    ).days + 1
    expected = EXPECTED_MONTHLY_LLM_CALLS * (days_remaining_window / days_in_month)
    if llm_calls >= 0.4 * expected and days_passed <= 0.25 * days_remaining_window:
        return f"FAST_BURN: {llm_calls} calls in {days_passed} days"
    return None


def alert_sustained_growth(usage: dict):
    daily = usage.get("daily_llm_calls", {})
    if len(daily) < 14:
        return None
    dates = sorted(daily.keys())
    last_7, prev_7 = dates[-7:], dates[-14:-7]
    avg_last = sum(daily[d] for d in last_7) / 7
    avg_prev = sum(daily[d] for d in prev_7) / 7
    if avg_last >= 2 * avg_prev:
        return f"SUSTAINED_GROWTH: {avg_last:.1f} vs {avg_prev:.1f}"
    return None


def alert_auto_block_proximity(usage: dict):
    if usage.get("llm_calls", 0) >= 0.75 * AUTO_BLOCK_MAX_LLM_CALLS_PER_MONTH:
        return f"AUTO_BLOCK_PROXIMITY: {usage['llm_calls']}"
    return None


def run_daily_monitoring():
    """Called by cron — not by the Flask app."""
    usage, f = load_monthly_usage()
    try:
        for check in [alert_daily_spike, alert_fast_monthly_burn,
                      alert_sustained_growth, alert_auto_block_proximity]:
            msg = check(usage)
            if msg:
                print(f"[ALERT] {msg}")
    finally:
        save_monthly_usage(f, usage, False)


# -----------------------
# INTERACTION STATE (PostgreSQL)
# -----------------------
"""
The two-step API flow (retrieve → predict) requires server-side state.
After /api/retrieve, the system stores retrieved_context, intent, confidence,
and all generation inputs in PostgreSQL under a UUID interaction_id.
The client gets back only the interaction_id — they cannot see or alter
the retrieved context. /api/predict fetches the state by interaction_id
and generates from it.

This design prevents prompt injection via tampered retrieved context,
and makes every generation fully auditable.
"""

def store_interaction_state(
    *,
    purpose: str,
    key_points: str,
    customer_context_summary: dict,
    workflow_type: str,
    ops_query: str,
    intent: str,
    intent_model_used: bool,
    intent_confidence: float,
    retrieved_context: str,
    retrieval_confidence: float,
    ttl_seconds: int,
):
    interaction_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds)

    for attempt in range(3):
        conn = get_pg_conn()
        try:
            conn.set_session(isolation_level=ISOLATION_LEVEL_SERIALIZABLE)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO interaction_state (
                        interaction_id, tenant_id, purpose, key_points,
                        customer_context_summary, workflow_type, ops_query,
                        intent, intent_model_used, intent_confidence,
                        retrieved_context, retrieval_confidence,
                        created_at, expires_at
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        interaction_id, "default", purpose, key_points,
                        Json(customer_context_summary), workflow_type, ops_query,
                        intent, intent_model_used, intent_confidence,
                        retrieved_context, retrieval_confidence,
                        now, expires_at,
                    )
                )
            conn.commit()
            return interaction_id

        except SerializationFailure:
            conn.rollback()
            if attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))
        finally:
            release_pg_conn(conn)


def get_interaction_state(*, interaction_id: str):
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT purpose, key_points, customer_context_summary,
                    workflow_type, ops_query, intent, intent_model_used,
                    intent_confidence, retrieved_context,
                    retrieval_confidence, expires_at
                FROM interaction_state
                WHERE interaction_id = %s AND tenant_id = %s
                """,
                (interaction_id, "default")
            )
            row = cur.fetchone()

        if not row:
            return None

        if row[-1] < datetime.now(timezone.utc):
            return None  # expired

        return {
            "purpose": row[0],
            "key_points": row[1],
            "customer_context_summary": row[2],
            "workflow_type": row[3],
            "ops_query": row[4],
            "intent": row[5],
            "intent_model_used": row[6],
            "intent_confidence": row[7],
            "retrieved_context": row[8],
            "retrieval_confidence": float(row[9]),
        }
    finally:
        release_pg_conn(conn)


# -----------------------
# INTERACTION AUDIT LOG
# -----------------------
def log_interaction_event(
    *,
    tenant_id: str,
    interaction_id: str,
    session_id: str | None,
    workflow_type: str,
    purpose: str,
    key_points: str,
    ops_query: str,
    intent: str,
    intent_model_used: bool,
    intent_confidence: float,
    retrieval_confidence: float,
    decision_outcome: str,
    generated_response: str | None,
):
    for attempt in range(3):
        conn = get_pg_conn()
        try:
            conn.set_session(isolation_level=ISOLATION_LEVEL_SERIALIZABLE)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO interaction_logs (
                        interaction_id, tenant_id, session_id, workflow_type,
                        purpose, key_points, ops_query, intent,
                        intent_model_used, intent_confidence,
                        retrieval_confidence, decision_outcome, generated_response
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        interaction_id, tenant_id, session_id, workflow_type,
                        purpose, key_points, ops_query, intent,
                        intent_model_used, intent_confidence,
                        retrieval_confidence, decision_outcome, generated_response,
                    )
                )
            conn.commit()
            break
        except SerializationFailure:
            conn.rollback()
            if attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))
        finally:
            release_pg_conn(conn)


# -----------------------
# DATABASE CONNECTION POOL
# -----------------------
# Connection pooling reuses existing PostgreSQL connections instead of
# opening a new TCP connection per request. Opening a connection takes
# ~20-50ms — at scale this dominates latency without pooling.
pg_pool = SimpleConnectionPool(minconn=1, maxconn=30, dsn=PG_DSN)

def get_pg_conn():
    return pg_pool.getconn()

def release_pg_conn(conn):
    pg_pool.putconn(conn)


def create_session_id() -> str:
    return str(uuid.uuid4())

def make_session_key(tenant_id: str, session_id: str) -> str:
    return f"{tenant_id}:{session_id}"


# -----------------------
# INPUT MASKING
# -----------------------
def sanitize_internal_input(text: str, mode: str = "strict") -> str:
    """
    Masks PII before storing in logs.
    We log inputs for audit purposes but must not store raw customer data —
    emails, phone numbers, card numbers, and tokens are replaced with
    placeholder tokens so logs are safe to retain and review.
    """
    if not text:
        return ""

    text = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"(?:\+?\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}", "[PHONE]", text)
    text = re.sub(r"\b(?:\d[ -]*?){13,19}\b", "[CARD]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"\b[A-Za-z0-9_\-]{20,}\b", "[TOKEN]", text)

    if mode == "strict":
        text = re.sub(r"\b\d{6,}\b", "[ID]", text)
        text = re.sub(r"\b[A-Z]{2,}[-_]\d{3,}\b", "[ID]", text)
        text = text[:500]

    elif mode == "relaxed":
        text = re.sub(r"\b\d{10,}\b", "[ID]", text)

    return text.strip()


# -----------------------
# KB LOADING + CHUNKING
# -----------------------
def chunk_large_text(text, max_len=500, overlap=80):
    """
    Splits text into overlapping chunks.
    Overlap preserves context across chunk boundaries — without it,
    a sentence split across two chunks would lose half its meaning
    in each chunk's embedding.

    Chunk 1: chars 0–500
    Chunk 2: chars 420–920  (80-char overlap)
    """
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + max_len
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def build_chunk(text, source, file_name, chunk_index):
    """
    Prepends metadata to each chunk so retrieval results are traceable.
    Format E.g.: [SOURCE:KB] [FILE:refund.txt] [CHUNK:3]\n<text>
    """
    return f"[SOURCE:{source}] [FILE:{file_name}] [CHUNK:{chunk_index}]\n{text}"


def load_and_chunk_text_file(path, fname, source="KB"):
    chunks = []
    try:
        with open(path, "rb") as f:
            raw = f.read()
        content = raw.decode("utf-8", errors="replace").strip()
    except Exception:
        safe_log_error("Failed to open KB file: %s", path)
        return chunks

    raw_chunks = chunk_large_text(content)
    seen_urls = set()

    for idx, c in enumerate(raw_chunks):
        expanded = expand_chunk_with_urls(
            build_chunk(c, source, fname, idx),
            seen_urls=seen_urls,
        )
        if expanded:
            chunks.extend(expanded)

    return chunks or []


def load_all_static_knowledge():
    """
    Loads from both permanent KB and uploads.
    Files with the same name in uploads/ override those in knowledge_base/
    — allowing clients to update KB without manual file replacement.
    """
    chunks = []

    uploads_files = set(f for f in os.listdir(UPLOADS_DIR) if f.lower().endswith(".txt"))
    kb_files = set(f for f in os.listdir(KB_DIR) if f.lower().endswith(".txt"))

    # uploads override KB for same filenames
    effective_kb_files = kb_files - uploads_files

    for fname in uploads_files:
        result = load_and_chunk_text_file(os.path.join(UPLOADS_DIR, fname), fname, source="UPLOADED_KB")
        if result:
            chunks.extend(result)

    for fname in effective_kb_files:
        result = load_and_chunk_text_file(os.path.join(KB_DIR, fname), fname, source="KB")
        if result:
            chunks.extend(result)

    url_list = load_url_list()
    url_chunks = load_url_knowledge(url_list)
    if url_chunks:
        chunks.extend(url_chunks)

    return chunks


# -----------------------
# URL SAFETY
# -----------------------
"""
The KB can reference URLs. Before fetching any URL, the system runs
SSRF (Server-Side Request Forgery) checks — blocking private IPs,
loopback addresses, and reserved ranges that a malicious KB file
could use to probe internal infrastructure.
"""

def extract_urls_from_text(text):
    return re.findall(r"(https?://[^\s]+)", text)


def dedupe_urls(urls):
    return list(set(urls))


def is_safe_url(url, allowed_domains=None):
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return False

        host = parsed.hostname
        if not host:
            return False

        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                return False
        except ValueError:
            pass  # host is a domain name, not an IP — allowed

        if allowed_domains:
            if not any(host.endswith(d) for d in allowed_domains):
                return False

        return True
    except Exception:
        return False


def host_is_public(host):
    try:
        infos = socket.getaddrinfo(host, None)
        for family, _, _, _, sockaddr in infos:
            ip = sockaddr[0]
            addr = ipaddress.ip_address(ip)
            if addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_link_local:
                return False
        return True
    except Exception:
        return False


def is_final_host_safe(url):
    BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"}
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        if not host or host in BLOCKED_HOSTS:
            return False
        return host_is_public(host)
    except Exception:
        return False


def safe_fetch_url(url):
    try:
        response = requests.get(url, timeout=(3, 5), stream=True, allow_redirects=True)

        content_length = int(response.headers.get("Content-Length", 0))
        if content_length > 2_000_000:
            return None

        if len(response.history) > 3:
            return None

        if not is_final_host_safe(response.url):
            return None

        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("text/") and "html" not in content_type:
            return None

        return response.text

    except Exception:
        safe_log_error("Failed to fetch URL: %s", url)
        return None


def load_url_knowledge(urls, allowed_domains=None):
    pages = []
    for url in urls:
        if not is_safe_url(url, allowed_domains):
            continue
        html = safe_fetch_url(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n").strip()
        file_name = url.replace("https://", "").replace("http://", "").replace("/", "_")
        for idx, chunk in enumerate(chunk_large_text(text)):
            pages.append(build_chunk(chunk, "URL", file_name, idx))
    return pages or []


def load_url_list():
    path = os.path.join(KB_DIR, "urls.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dedupe_urls(data.get("urls", []))
    except Exception:
        safe_log_error("Failed to load URLs list.")
        return []


def expand_chunk_with_urls(original_chunk, allowed_domains=None, seen_urls=None):
    """
    If a KB chunk contains URLs, fetches and appends their content.
    seen_urls prevents the same URL from being fetched multiple times
    across different chunks in the same indexing pass.
    """
    urls = dedupe_urls(extract_urls_from_text(original_chunk))
    expanded = [original_chunk]

    for url in urls:
        if seen_urls is not None:
            if url in seen_urls:
                continue
            seen_urls.add(url)

        url_chunks = load_url_knowledge([url], allowed_domains)
        if url_chunks:
            expanded.extend(url_chunks)

    return expanded or []


# -----------------------
# KB INDEX BUILDING
# -----------------------
def ensure_kb_index(bert_model):
    """
    Loads the FAISS KB index if it exists.
    Builds and saves it from scratch if it doesn't.
    This happens once at first request — subsequent requests load
    from disk, which takes ~50ms vs minutes of re-embedding.
    """
    if faiss is None:
        return None, []

    index, metadata = load_faiss_index(KB_INDEX_PATH, KB_META_PATH)
    if index is not None:
        return index, metadata

    kb_chunks = load_all_static_knowledge()
    kb_chunks = dedupe_chunks_by_text(kb_chunks)
    kb_chunks = dedupe_substrings(kb_chunks)
    kb_chunks = semantic_dedupe(kb_chunks, bert_model)

    if not kb_chunks:
        return None, []

    kb_embeddings = bert_model.encode(kb_chunks, convert_to_numpy=True).astype("float32")

    dim = kb_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(kb_embeddings)

    save_faiss_index(index, KB_INDEX_PATH, KB_META_PATH, kb_chunks)
    return index, kb_chunks


def ingest_uploaded_files_with_metadata(file_path):
    final_chunks = []
    file_name = os.path.basename(file_path)
    text = ""

    if file_name.lower().endswith(".txt"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception:
            safe_log_error("Failed to open uploaded KB file: %s", file_path)

    raw_chunks = chunk_large_text(text)
    seen_urls = set()

    for idx, c in enumerate(raw_chunks):
        expanded = expand_chunk_with_urls(
            build_chunk(c, "UPLOADED_KB", file_name, idx),
            seen_urls=seen_urls,
        )
        if expanded:
            final_chunks.extend(expanded)

    return final_chunks


def index_uploaded_file(file_path, bert_model):
    """
    Incrementally adds a new file to the uploads FAISS index.
    Loads the existing index first so previous uploads are preserved.
    Vector count cap prevents unbounded memory growth.
    """
    if faiss is None:
        return {"status": "error", "message": "faiss_not_available"}

    if not isinstance(file_path, str):
        return {"status": "error", "message": "invalid_file_path"}

    up_index, up_meta = load_faiss_index(UP_INDEX_PATH, UP_META_PATH)

    if up_index is None:
        up_meta = []
        emb = bert_model.encode(["text"], convert_to_numpy=True).astype("float32")
        up_index = faiss.IndexFlatL2(emb.shape[1])

    chunks = ingest_uploaded_files_with_metadata(file_path)
    chunks = dedupe_chunks_by_text(chunks)
    chunks = dedupe_substrings(chunks)
    chunks = semantic_dedupe(chunks, bert_model)

    if not chunks:
        return {"status": "error", "message": "no_valid_chunks"}

    embeddings = bert_model.encode(chunks, convert_to_numpy=True).astype("float32")

    if up_index.ntotal + len(embeddings) > MAX_VECTORS:
        return {"status": "error", "message": f"vector_limit_exceeded: max={MAX_VECTORS}"}

    up_index.add(embeddings)
    up_meta.extend(chunks)

    save_faiss_index(up_index, UP_INDEX_PATH, UP_META_PATH, up_meta)
    return {"status": "success"}


# -----------------------
# DEDUPLICATION
# -----------------------
def dedupe_chunks_by_text(chunk_list):
    """Removes chunks with identical text content. Keeps overlapping chunks."""
    seen = set()
    deduped = []
    for chunk in chunk_list:
        parts = chunk.split("\n", 1)
        text = parts[1].strip() if len(parts) == 2 else chunk
        if text not in seen:
            seen.add(text)
            deduped.append(chunk)
    return deduped


def dedupe_substrings(chunks):
    """
    Removes chunk A if its entire text appears inside chunk B.
    Preserves overlapping chunks where neither is a full substring of the other.
    """
    final = []
    for i, c1 in enumerate(chunks):
        text1 = c1.split("\n", 1)[1].strip() if "\n" in c1 else c1
        keep = True
        for j, c2 in enumerate(chunks):
            if i != j:
                text2 = c2.split("\n", 1)[1].strip() if "\n" in c2 else c2
                if text1 in text2 and len(text1) < len(text2):
                    keep = False
                    break
        if keep:
            final.append(c1)
    return final


def semantic_dedupe(chunks, bert_model, threshold=0.96):
    """
    Removes semantically near-duplicate chunks using cosine similarity.
    When two chunks are above the threshold, the higher-authority source wins
    (UPLOADED_KB > TENANT_KB > KB > URL). This means client-uploaded KB
    always takes precedence over the default knowledge base for the same topic.
    """
    if not chunks:
        return chunks

    texts = [c.split("\n", 1)[1] if "\n" in c else c for c in chunks]
    embeddings = bert_model.encode(texts, convert_to_numpy=True)

    SOURCE_PRIORITY = {
        "UPLOADED_KB": 3,
        "TENANT_KB": 2,
        "KB": 1,
        "URL": 0,
    }

    keep = [True] * len(chunks)
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if keep[i] and keep[j]:
                sim = cos_sim(embeddings[i], embeddings[j]).item()
                if sim > threshold:
                    src_i = chunks[i].split("\n", 1)[0]
                    src_j = chunks[j].split("\n", 1)[0]

                    if src_i == src_j:
                        keep[j if len(texts[i]) >= len(texts[j]) else i] = False
                        continue

                    pri_i = next((v for k, v in SOURCE_PRIORITY.items() if k in src_i), 0)
                    pri_j = next((v for k, v in SOURCE_PRIORITY.items() if k in src_j), 0)

                    if pri_i > pri_j:
                        keep[j] = False
                    elif pri_j > pri_i:
                        keep[i] = False
                    else:
                        keep[j if len(texts[i]) >= len(texts[j]) else i] = False

    return [chunks[i] for i in range(len(chunks)) if keep[i]]


# -----------------------
# RETRIEVAL CONFIDENCE
# -----------------------
def compute_retrieval_confidence(scores):
    """
    Weighted confidence score from three retrieval signals.

    All three signals are linearly mapped to [0, 1] before weighting:
    - Strength:      how good is the best match in absolute terms?
    - Separation:    how much better is rank-1 vs rank-2?
    - Concentration: how much better is rank-1 vs the average of the rest?

    Weights (0.5 / 0.3 / 0.2) reflect that absolute quality matters
    more than separation, which matters more than concentration.

    Assumes scores are post normalize_l2_distance(), i.e., in 
    range of [0, 1).

    Although the mathematical range approaches 1.0,
    the confidence system treats scores near ~0.8 as
    practically weak retrievals for calibration purposes.
    That's why 0.8 is being used for calibrating.
    """
    if not scores:
        return 0.0

    scores = sorted(scores)
    top1 = scores[0]

    if len(scores) > 1:
        gap      = scores[1] - scores[0]
        avg_rest = sum(scores[1:]) / (len(scores) - 1)
    else:
        # Only one result — separation and concentration are undefined.
        # Confidence is driven by strength alone, capping at 0.5.
        gap      = 0.0
        avg_rest = top1

    # STRENGTH — how good is the best match in absolute terms?
    # top1 is practically expected to fall near the 0.0–0.8 range for useful retrieval matches.
    # Low distance = high strength.
    # Weight: 0.5 — primary signal.
    strength = 1.0 - (top1 / 0.8)

    # SEPARATION — how much better is rank 1 than rank 2?
    # Gap values are calibrated relative to the practical 0.0–0.8 retrieval-quality range.
    # Weight: 0.3 — secondary signal.
    separation = gap / 0.8

    # CONCENTRATION — how much better is rank 1 than the average of the rest?
    # Concentration is calibrated relative to the practical 0.0–0.8 retrieval-quality range.
    # Large value means relevance is focused on rank 1, not spread across results.
    # Weight: 0.2 — supporting signal.
    concentration = (avg_rest - top1) / 0.8

    retrieval_confidence = (
        0.5 * strength +
        0.3 * separation +
        0.2 * concentration
    )

    # Clamp to [0.0, 1.0] as a safeguard against floating point edge cases.
    return max(0.0, min(retrieval_confidence, 1.0))



# -----------------------
# L2 DISTANCE NORMALIZATION
# -----------------------
def normalize_l2_distance(d):
    """
    Compresses FAISS squared L2 distances into a smoother bounded range
    for easier threshold comparison and confidence computation.

    FAISS IndexFlatL2 returns squared L2 distances.

    For unit-normalized embeddings, squared L2 distance relates to cosine
    similarity as:

        squared_L2 = 2 - 2 * cosine_similarity

    which gives a theoretical range of [0, 4] (only when embeddings are
    normalized). If embeddings are not normalized, then the range is
    of [0, ∞). 

    This function applies:

        d / (d + 1)

    to compress distances into the range [0, 1), while preserving
    ranking order (smaller distance = higher similarity).
    """
    return d / (d + 1)


# -----------------------
# MAIN RETRIEVAL FUNCTION
# -----------------------
def retrieve_context(bert_model, query_text: str):
    """
    Dual-index retrieval: searches permanent KB and uploads separately,
    then merges and re-ranks by distance.

    Searching separately (not a merged index) allows different top_k
    budgets per source — uploads get fewer slots (3) because they're
    typically more targeted, KB gets more (5) for broader coverage.
    Final top_k = 6 from the merged + sorted combined results.
    """
    kb_top_k = 5
    up_top_k = 3
    final_top_k = 6

    kb_index, kb_meta = ensure_kb_index(bert_model)
    up_index, up_meta = load_faiss_index(UP_INDEX_PATH, UP_META_PATH)

    q_vec = bert_model.encode([query_text], convert_to_numpy=True).astype("float32")

    results = []

    if kb_index is not None and kb_meta:
        kb_k = min(kb_top_k, len(kb_meta))
        kb_dist, kb_inds = kb_index.search(q_vec, kb_k)
        for i in range(kb_k):
            idx = kb_inds[0][i]
            if 0 <= idx < len(kb_meta):
                results.append((normalize_l2_distance(float(kb_dist[0][i])), kb_meta[idx]))

    if up_index is not None and up_meta:
        up_k = min(up_top_k, len(up_meta))
        up_dist, up_inds = up_index.search(q_vec, up_k)
        for i in range(up_k):
            idx = up_inds[0][i]
            if 0 <= idx < len(up_meta):
                results.append((normalize_l2_distance(float(up_dist[0][i])), up_meta[idx]))

    if not results:
        return "", [], 0.0, [], []

    results.sort(key=lambda x: x[0])
    final = results[:final_top_k]

    similarity_scores = [r[0] for r in final]
    chunks = [r[1] for r in final]
    retrieval_confidence = compute_retrieval_confidence(similarity_scores)
    sources_used = [c.split("\n", 1)[0] for c in chunks]
    context_text = "\n---\n".join(chunks)

    debug_results = [
        {
            "rank": rank + 1,
            "score": score,
            "source": chunk.split("\n", 1)[0],
            "chunk_preview": chunk.split("\n", 1)[1][:200] if "\n" in chunk else chunk[:200],
        }
        for rank, (score, chunk) in enumerate(final)
    ]

    return context_text, similarity_scores, retrieval_confidence, sources_used, debug_results


# -----------------------
# MISC HELPERS
# -----------------------
def safe_csv(val):
    """Prevents CSV injection — Excel treats cells starting with =,+,-,@ as formulas."""
    if isinstance(val, str) and val.startswith(("=", "+", "-", "@")):
        return "'" + val
    return val
