import sys
import shutil
import logging
import tempfile
import os
import re
import json
import hmac
import hashlib
import threading
import queue
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from functools import wraps

from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import uuid

load_dotenv()

from main_logic import (
    load_bert_model,
    retrieve_context,
    predict_intent,
    sanitize_internal_input,
    get_pg_conn,
    log_interaction_event,
    get_all_kb_files,
    cleanup_orphan_temp_dirs,
    release_pg_conn,
    safe_log_error,
    safe_log_info,
    safe_log_warning,
    store_interaction_state,
    get_interaction_state,
    sanitize_filename,
    api_error,
    index_uploaded_file,
    rebuild_index,
    clear_uploads_vector_store,
    extract_text_from_file,
    sanitize_kb_text,
    store_cleaned_kb_text,
    compute_sha256_bytes,
    store_raw_kb_file,
    validate_kb_filename,
    create_secure_temp_dir,
    validate_file_size_for_temp,
    record_llm_usage,
    record_request_usage,
    enforce_auto_block_precheck,
    record_security_requests,
    record_security_llm_call
)

APP_START_TIME = time.time()


os.environ["OMP_NUM_THREADS"] = "1" # Limits OpenMP-based libraries (NumPy, PyTorch, FAISS) to use only 1 CPU thread per request. Prevents thread explosion and keeps API stable under concurrent traffic.
os.environ["MKL_NUM_THREADS"] = "1" # Restricts Intel MKL (used by NumPy/sklearn) to a single thread for matrix computations. Avoids sudden CPU spikes during embedding and intent prediction.
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disables multi-threading in HuggingFace tokenizers used inside SentenceTransformer and pipelines. Prevents warnings, deadlocks, and unstable behavior in a multi-request server.


# -----------------------
# LOGGING
# -----------------------
# Named logger shared across all modules
# logging.getLogger("rag_system") to attach to the same handler.
logger = logging.getLogger("rag_system")
logger.setLevel(logging.INFO)
logger.propagate = False  # prevents double-printing via root logger

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
os.makedirs(KB_DIR, exist_ok=True)

MAX_KB_FILES = 50

# Confidence gate: if retrieval confidence falls below this,
# the system escalates instead of generating.
# Wrong answers in high-risk workflows are worse than no answer.
MIN_RETRIEVAL_CONFIDENCE = 0.35
MAX_ALLOWED_BEST_DISTANCE = 0.55

MAX_SKEW_SECONDS = 300  # replay attack protection window

ALLOWED_ATTACHMENT_EXTENSIONS = {".pdf", ".txt"}
MAX_SINGLE_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_ATTACHMENTS = 5

MAX_PAST_EMAIL_CHARS = 5000
MAX_SINGLE_ATTACHMENT_CHARS = 4000
MAX_TOTAL_TEXT_CHARS = 12000

MAX_KEY_POINTS_CHARS = 1000
MAX_PURPOSE_CHARS = 300

# Per-endpoint rate limits (requests per minute per API key).
# These are calibrated so legitimate high-volume clients
# don't hit them under normal usage patterns.
RATE_LIMITS = {
    "api_retrieve": 120,
    "api_predict": 60,
    "api_ingest_kb": 10,
    "api_feedback": 30,
    "api_health": 300,
}

# Per-IP hourly abuse thresholds — separate from rate limiting.
# Rate limiting protects cost; abuse detection protects integrity.
ABUSE_THRESHOLD_PER_ENDPOINT = {
    "api_retrieve": 10000,
    "api_predict": 6000,
    "api_feedback": 2000,
    "api_ingest_kb": 850,
}

# Intents that always escalate for high-risk clients,
# regardless of retrieval confidence.
HARD_ESCALATE_INTENTS = {
    "refund_request",
    "complaint",
}

# Keywords that bypass intent classification and force escalation.
# These cover legal/financial risk signals that classifiers can miss.
HARD_ESCALATE_KEYWORDS = [
    "legal", "lawyer", "lawsuit", "court", "gdpr", "privacy",
    "compliance", "regulation", "chargeback", "bank",
    "payment dispute", "credit card", "refund immediately",
    "cancel immediately", "terminate account", "sue",
]

INTERACTION_TTL = timedelta(hours=2)

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

SECURITY_ROOT = os.path.join(BASE_DIR, "security")
os.makedirs(SECURITY_ROOT, exist_ok=True)

IP_HASH_SALT = os.environ["IP_HASH_SALT"]
if not IP_HASH_SALT:
    raise RuntimeError("IP_HASH_SALT must be set in environment")

rate_limit_store = defaultdict(lambda: deque())
ip_abuse_store = defaultdict(lambda: deque())

TEMP_KB_DIR_PATH = os.path.join(BASE_DIR, "tmp/uploads")
os.makedirs(TEMP_KB_DIR_PATH, exist_ok=True)
cleanup_orphan_temp_dirs(TEMP_KB_DIR_PATH)


# -----------------------
# SYSTEM PROMPTS
# -----------------------

SYSTEM_PROMPT_FOR_SUPPORT_ASSISTANCE = """
You are a summarization assistant inside a production AI system.

Your task is to extract structured information from input with high factual fidelity.

STRICT RULES:
- Return ONLY valid JSON
- Do NOT include any text outside JSON
- Do NOT add explanations
- Do NOT infer or assume anything
- Preserve all factual details exactly as stated
- Include every distinct critical fact that may matter for retrieval or downstream decision-making

OUTPUT FORMAT (MANDATORY):

{
  "summary": "One or two concise sentences capturing the customer's core request, issue, or intent",
  "key_details": [
    "critical fact"
  ]
}

SUMMARY RULES:
- Keep it concise
- Capture the core request, issue, or intent
- Do not omit important context

KEY_DETAILS RULES:
- key_details is a variable-length array
- Include ALL distinct critical facts that appear in the input
- Each item must contain ONLY ONE fact
- Use short factual phrases
- Prioritize facts affecting eligibility, decisions, outcomes, timing, amounts, conditions, or entities
- Include dates, durations, amounts, quantities, status, conditions, identifiers, and other decision-relevant facts
- Do NOT merge multiple facts into one item
- Do NOT repeat the same fact in different words
- Do NOT add generic filler items
- If no key details exist, return []


PRIORITY RULE:
- If both conversation and attachments contain facts, preserve facts from both
- If the attachment contains more precise factual information, prefer it over less precise conversation wording
"""


# -----------------------
# ASYNC DISK WRITE QUEUE
# -----------------------
# Disk writes (log files) are queued to a background thread.
# This removes file I/O from the request critical path.
log_queue = queue.Queue()

def log_writer():
    while True:
        log_file_path, log_data = log_queue.get()
        try:
            with open(log_file_path, "a") as f:
                f.write(log_data + "\n")
        except Exception:
            safe_log_error("Failed to write log to '%s'.", log_file_path)
        finally:
            log_queue.task_done()

threading.Thread(target=log_writer, daemon=True).start()


# -----------------------
# LOGGING HELPERS
# -----------------------
def log_api_predict_results(tenant_id, request_id, workflow_type, purpose,
                             key_points, customer_context_summary, ops_query,
                             intent, generated_reply, latency_ms):
    log = {
        "tenant_id": tenant_id,
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workflow_type": workflow_type,
        "purpose": purpose,
        "key_points": key_points,
        "customer_context_summary": customer_context_summary,
        "ops_query": ops_query,
        "intent": intent,
        "generated_reply": generated_reply,
        "latency_ms": latency_ms,
    }
    log_queue.put((os.path.join(LOG_DIR, "api_predict_results.log"), json.dumps(log)))


def log_api_event(tenant_id, endpoint, status, latency_ms, ip_hash, request_id, user_agent):
    log = {
        "tenant_id": tenant_id,
        "request_id": request_id,
        "endpoint": endpoint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "latency_ms": latency_ms,
        "ip_hash": ip_hash,
        "user_agent": user_agent,
    }
    log_queue.put((os.path.join(LOG_DIR, "api_requests.log"), json.dumps(log)))


def log_escalation(*, tenant_id, request_id, endpoint, escalation_reason, ip_hash):
    log = {
        "tenant_id": tenant_id,
        "request_id": request_id,
        "endpoint": endpoint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "escalation_reason": escalation_reason,
        "ip_hash": ip_hash,
    }
    log_queue.put((os.path.join(LOG_DIR, "escalation.log"), json.dumps(log)))


def log_kb_ingestion(tenant_id, status, filename, request_id, error):
    log = {
        "tenant_id": tenant_id,
        "request_id": request_id,
        "status": status,
        "filename": filename,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_queue.put((os.path.join(LOG_DIR, "kb_ingestion_status.log"), json.dumps(log)))


def log_intent_model_status(tenant_id, interaction_id, intent,
                             intent_model_used, intent_confidence, intent_status):
    log = {
        "tenant_id": tenant_id,
        "interaction_id": interaction_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intent": intent,
        "intent_model_used": intent_model_used,
        "intent_confidence": intent_confidence,
        "intent_status": intent_status,
    }
    log_queue.put((os.path.join(LOG_DIR, "intent_model_status.log"), json.dumps(log)))


def log_debug_retrieval_results(tenant_id, ip_hash, request_id,
                                 interaction_id, retrieval_results):
    log = {
        "tenant_id": tenant_id,
        "request_id": request_id,
        "interaction_id": interaction_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ip_hash": ip_hash,
        "retrieval_results": retrieval_results,
    }
    log_queue.put((os.path.join(LOG_DIR, "api_retrieval_results.log"), json.dumps(log)))


# -----------------------
# KB INGESTION
# -----------------------
def ingest_kb_file(*, uploaded_file, request_id, safe_filename,
                   raw_bytes, client_hash=None, overwrite):
    safe_filename = sanitize_filename(safe_filename)
    server_hash = compute_sha256_bytes(raw_bytes)

    # Optional client-side hash verification — prevents disputes
    # about whether the file we stored matches what they sent.
    if client_hash and client_hash != server_hash:
        return {"status": "error", "message": "Hash mismatch", "status_code": 400}

    raw_path = store_raw_kb_file(request_id=request_id, filename=safe_filename, raw_bytes=raw_bytes)
    temp_dir = create_secure_temp_dir(request_id)
    cleaned_text = ""

    try:
        validate_result = validate_kb_filename(safe_filename)
        if validate_result["status"] == "rejected":
            return {
                "status": "rejected",
                "message": validate_result["message"],
                "message_for_client": validate_result["message_for_client"],
                "status_code": validate_result["status_code"],
                "code": validate_result["code"],
            }

        temp_path = os.path.join(temp_dir, safe_filename)
        uploaded_file.save(temp_path)

        try:
            validate_file_size_for_temp(temp_path)
        except ValueError as e:
            return {"status": "error", "message": str(e), "status_code": 400}

        try:
            raw_text = extract_text_from_file(temp_path)
        except ValueError as e:
            return {"status": "error", "message": str(e), "status_code": 400}
        except Exception:
            return {"status": "error", "message": "exception"}

        try:
            cleaned_text = sanitize_kb_text(raw_text)
        except ValueError as e:
            return {"status": "error", "message": str(e), "status_code": 400}

        cleaned_path = store_cleaned_kb_text(
            safe_filename=safe_filename,
            cleaned_text=cleaned_text,
            overwrite=overwrite,
        )

        log_kb_ingestion(
            tenant_id="default",
            request_id=request_id,
            status="kb_ingestion_succeed",
            filename=safe_filename,
            error="None",
        )

        return {"status": "success", "cleaned_path": cleaned_path, "raw_path": raw_path}

    except Exception as e:
        log_kb_ingestion(
            tenant_id="default",
            request_id=request_id,
            status="kb_ingestion_failed",
            filename=safe_filename,
            error=str(e),
        )
        safe_log_error("Unexpected exception inside 'ingest_kb_file()'.")
        return {"status": "error", "message": "KB ingestion failed", "status_code": 500}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------
# ATTACHMENT PROCESSING
# -----------------------
def is_allowed_file(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_ATTACHMENT_EXTENSIONS


def validate_file_size(file_obj):
    file_obj.seek(0, os.SEEK_END)
    size = file_obj.tell()
    file_obj.seek(0)
    if size > MAX_SINGLE_FILE_BYTES:
        return {"status": "error", "message": "File too large. Max: 5MB"}
    return {"status": "success"}


def extract_text_from_txt(file_obj):
    file_obj.seek(0)
    raw = file_obj.read()
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        return {"text": "", "status": "error", "message": str(e)}
    return {"text": text.strip(), "status": "success", "message": ""}


def extract_text_from_attached_pdf(file_obj):
    file_obj.seek(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        safe_name = sanitize_filename(file_obj.filename)
        path = os.path.join(tmpdir, safe_name)
        with open(path, "wb") as f:
            f.write(file_obj.read())
        text_chunks = []
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_chunks.append(t)
        except Exception as e:
            return {"text": "", "status": "error", "message": str(e)}
    return {"text": "\n".join(text_chunks).strip(), "status": "success", "message": ""}


def process_attachments(files):
    if not files:
        yield {"status": "success", "text": "empty"}
        return

    if len(files) > MAX_TOTAL_ATTACHMENTS:
        yield {"status": "error", "message": "Too many attachments"}
        return

    for file_obj in files:
        filename = sanitize_filename(file_obj.filename)

        if not is_allowed_file(filename):
            yield {"status": "error", "message": "Unsupported file type. Allowed: .txt, .pdf"}
            return

        result = validate_file_size(file_obj)
        if result["status"] == "error":
            yield result
            return

        if filename.endswith(".txt"):
            r = extract_text_from_txt(file_obj)
            if r["status"] == "error":
                safe_log_warning("txt extraction failed: %s", r["message"])
            text = r["text"]
        elif filename.endswith(".pdf"):
            r = extract_text_from_attached_pdf(file_obj)
            if r["status"] == "error":
                safe_log_warning("pdf extraction failed: %s", r["message"])
            text = r["text"]
        else:
            continue

        if text.strip():
            yield {"status": "success", "text": text}


# -----------------------
# TEXT CLEANING
# -----------------------
def clean_customer_text(raw_text):
    """
    Cleans past email threads or attachment text.
    Strips quoted replies, thread headers, signatures, and disclaimers
    so only the customer's most recent message reaches the LLM.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()
    text = re.sub(r"(?m)^>.*$", "", text)

    for pattern in [
        r"On .* wrote:", r"From:.*", r"Sent:.*",
        r"To:.*", r"Subject:.*",
        r"-----Original Message-----", r"----- Forwarded message -----",
    ]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    for pattern in [
        r"(?s)--\s*\n.*$", r"(?s)Thanks[,]?\n.*$",
        r"(?s)Best regards[,]?\n.*$", r"(?s)Regards[,]?\n.*$",
        r"(?s)Sincerely[,]?\n.*$",
    ]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    for pattern in [
        r"(?s)this email.*confidential.*", r"(?s)this message.*confidential.*",
        r"(?s)intended solely for.*",
    ]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate_text(text, max_chars):
    if not text:
        return ""
    return text[:max_chars]


def process_past_email(raw_email):
    return truncate_text(clean_customer_text(raw_email), MAX_PAST_EMAIL_CHARS)


def process_attachment_text(raw_text):
    return truncate_text(clean_customer_text(raw_text), MAX_SINGLE_ATTACHMENT_CHARS)


def is_text_length_long(combined_customer_text, past_email_len):
    if not combined_customer_text:
        return "", False, "", 0, 0

    attachments_text_len = len(combined_customer_text) - past_email_len

    if len(combined_customer_text) > MAX_TOTAL_TEXT_CHARS:
        reason = "past_email_length_too_long" if past_email_len > 3000 else "attachment(s)_text_length_too_long"
        return (combined_customer_text[:MAX_TOTAL_TEXT_CHARS], True,
                reason, past_email_len, attachments_text_len)

    return combined_customer_text, False, "", past_email_len, attachments_text_len


# -----------------------
# SUMMARY GENERATION
# -----------------------
def extract_json_block(text):
    start = text.find("{")
    if start == -1:
        return None
    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i + 1]
    return None


def fallback_parse_summary(raw_output):
    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
    if not lines:
        return {"summary": "", "key_details": []}
    summary = lines[0]
    key_details = [re.sub(r"^[-•\d\.\)]\s*", "", l).strip() for l in lines[1:] if l.strip()]
    return {"summary": summary, "key_details": key_details[:7]}


def generate_summary(combined_text):
    """
    Extracts a structured summary + key facts from customer input.
    Using structured JSON output from the LLM here rather than free text
    because downstream retrieval quality depends on precise factual extraction,
    not loosely rewritten responses.
    """

    if not combined_text:
        return {"summary": "", "key_details": []}

    from llm_client import get_openai_client
    client = get_openai_client()

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_FOR_SUPPORT_ASSISTANCE.strip()},
                    {"role": "user", "content": f"INPUT:\n\n{combined_text}"},
                ],
                temperature=0.0,
                max_tokens=200,
                timeout=30,
                top_p=1,
            )

            raw_output = response.choices[0].message.content.strip()
            if not raw_output:
                raise ValueError("Empty summary output")

            try:
                parsed = json.loads(raw_output)
            except Exception:
                json_text = extract_json_block(raw_output)
                if json_text:
                    try:
                        parsed = json.loads(json_text)
                    except Exception:
                        return fallback_parse_summary(raw_output)
                else:
                    return fallback_parse_summary(raw_output)

            summary = parsed.get("summary", "").strip()
            key_details = parsed.get("key_details", [])

            if not isinstance(summary, str) or not summary:
                raise ValueError("Invalid summary")
            if not isinstance(key_details, list):
                raise ValueError("key_details must be a list")

            key_details = [str(x).strip() for x in key_details if str(x).strip()][:7]
            return {"summary": summary, "key_details": key_details}

        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"LLM summary failed: {str(e)}")
            time.sleep(1 * (attempt + 1))


# -----------------------
# QUERY BUILDING
# -----------------------
def clean_purpose(purpose):
    return re.sub(r"\s+", " ", purpose.strip())[:MAX_PURPOSE_CHARS] if purpose else ""


def clean_key_points(key_points):
    return re.sub(r"\s+", " ", key_points.strip())[:MAX_KEY_POINTS_CHARS] if key_points else ""


def build_retrieval_query(*, key_points, customer_context_summary=None, ops_query=""):
    """
    Builds the semantic query for FAISS vector search.
    Order: summary > key_details > key_points for support workflows.
    ops_query > key_points for internal/financial/compliance workflows.
    Purpose is intentionally excluded — it describes intent, not content.
    """
    parts = []

    if customer_context_summary:
        summary = customer_context_summary["summary"]
        key_details = [d.strip() for d in customer_context_summary.get("key_details", []) if d.strip()]
        parts.append(summary)
        if key_details:
            parts.append(" ".join(key_details))
        if key_points:
            parts.append(key_points.strip())
        return " ".join(parts)

    if ops_query:
        parts.append(ops_query)
    if key_points:
        parts.append(key_points.strip())
    return " ".join(parts)


# -----------------------
# LLM GENERATION
# -----------------------
def generate_support_draft(purpose, summary, retrieved_context, key_points=None):
    if not summary or not retrieved_context:
        raise ValueError("Missing required inputs for support generation")

    from llm_client import get_openai_client
    client = get_openai_client()

    SYSTEM_PROMPT = """
You are an AI assistant operating inside a governed production system for support agent assistance.

Your job is to draft a customer-facing email response that the support agent can send to the customer.

STRICT RULES:
- Use ONLY the provided retrieved knowledge and the provided summary
- Do NOT add assumptions, extra facts, or outside knowledge
- Do NOT invent policies, promises, timelines, or outcomes
- If the answer is not clearly supported, respond conservatively and do not guess
- Keep the response professional, clear, and customer-friendly

STYLE:
- Generate ONLY the email body
- Do NOT include subject line, email headers, greetings, or sign-offs
"""

    USER_PROMPT = f"""
Purpose:
{purpose}

Key Points (may be empty):
{key_points}

Customer Summary:
{summary}

Retrieved Knowledge:
{retrieved_context}

Write the customer-facing email response body.
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": USER_PROMPT.strip()},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=250,
                timeout=30,
            )
            output = response.choices[0].message.content.strip()
            if not output:
                raise ValueError("Empty generation output")
            return output
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Support draft generation failed: {str(e)}")
            time.sleep(1 * (attempt + 1))


def generate_ops_response(purpose, ops_query, retrieved_context, workflow_type, key_points=None):
    if not ops_query or not retrieved_context:
        raise ValueError("Missing required inputs for ops generation")

    from llm_client import get_openai_client
    client = get_openai_client()

    SYSTEM_PROMPTS = {
        "internal_ops": """
You are an AI assistant operating inside a governed production system.
Use ONLY the provided retrieved knowledge. Do NOT make decisions or approvals.
Be concise and factual. If the answer is not in the retrieved knowledge, say so.
""",
        "financial_ops": """
You are an AI assistant for finance and revenue operations.
Use ONLY the provided retrieved knowledge. Do NOT approve or reject financial actions.
Do NOT invent policies, limits, or timelines. Be concise and operational.
""",
        "compliance_validation": """
You are an AI assistant for compliance and policy validation.
Present what is explicitly stated in the policy. Do NOT interpret or conclude compliance.
Do NOT say "allowed", "not allowed", "compliant", or "non-compliant". Present facts only.
""",
    }

    SYSTEM_PROMPT = SYSTEM_PROMPTS.get(workflow_type, SYSTEM_PROMPTS["internal_ops"])

    USER_PROMPT = f"""
Purpose:
{purpose}

Key Constraints (can be empty):
{key_points or "None"}

User Query:
{ops_query}

Retrieved Knowledge:
{retrieved_context}
"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": USER_PROMPT.strip()},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=270,
                timeout=30,
            )
            output = response.choices[0].message.content.strip()
            if not output:
                raise ValueError("Empty generation output")
            return output
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Ops response generation failed: {str(e)}")
            time.sleep(1 * (attempt + 1))


# -----------------------
# ESCALATION LOGIC
# -----------------------
def should_escalate(allow_hard_escalate, intent, customer_context_summary,
                    ops_query, best_distance, retrieval_confidence):
    """
    Escalation runs before generation — not after.
    The system would rather surface uncertainty to a human
    than generate a confident wrong answer in a high-risk workflow.
    """
    if allow_hard_escalate and intent and intent in HARD_ESCALATE_INTENTS:
        return True, f"hard_intent: {intent}"

    if retrieval_confidence < MIN_RETRIEVAL_CONFIDENCE:
        return True, f"retrieval_confidence {retrieval_confidence} below threshold"

    if best_distance is None or best_distance > MAX_ALLOWED_BEST_DISTANCE:
        return True, f"best_distance {best_distance} exceeds threshold"

    lowered = (customer_context_summary["summary"] if customer_context_summary else ops_query or "").lower()

    if allow_hard_escalate:
        for keyword in HARD_ESCALATE_KEYWORDS:
            if keyword in lowered:
                return True, f"keyword: {keyword}"

    return False, ""


# -----------------------
# SECURITY: IP HASHING + RATE LIMITING + ABUSE DETECTION
# -----------------------
def get_client_ip():
    return request.remote_addr or "unknown"


def hash_ip(ip):
    # IPs are hashed before logging — raw IPs are never stored.
    # Salt prevents rainbow table attacks against the hash.
    h = hashlib.sha256()
    h.update(f"{IP_HASH_SALT}:{ip}".encode())
    return h.hexdigest()


def enforce_rate_limit(api_key, endpoint):
    """
    Sliding window rate limiter — tracks request timestamps per
    (api_key, endpoint) pair in a deque and evicts entries older
    than 60 seconds. O(1) amortized per request.
    """
    limit = RATE_LIMITS.get(endpoint)
    if not limit:
        return True

    now = time.time()
    window = 60
    q = rate_limit_store[(api_key, endpoint)]

    while q and q[0] <= now - window:
        q.popleft()

    if len(q) >= limit:
        return False

    q.append(now)
    return True


def record_ip_abuse_signal(ip_hash, endpoint):
    """
    Separate from rate limiting — rate limiting gates per API key,
    abuse detection gates per IP. An attacker rotating keys would
    still be caught here.
    """
    limit = ABUSE_THRESHOLD_PER_ENDPOINT.get(endpoint)
    if not limit:
        return True

    now = time.time()
    window = 3600
    q = ip_abuse_store[(ip_hash, endpoint)]

    while q and q[0] <= now - window:
        q.popleft()

    if len(q) >= limit:
        return False

    q.append(now)
    return True


# -----------------------
# HMAC REQUEST SIGNING
# -----------------------
def verify_hmac_signature(secret, method, path, timestamp, body, signature):
    """
    Each request is signed with HMAC-SHA256 over method + path + timestamp + body hash.

    This helps ensure:
    - the request body was not modified in transit
    - old captured requests cannot be replayed later
    - possession of an API key alone is not enough without the signing secret
    """
    
    try:
        now = int(time.time())
        ts = int(timestamp)
    except Exception:
        return False

    if abs(now - ts) > MAX_SKEW_SECONDS:
        return False

    body_hash = hashlib.sha256(body).hexdigest() if body else ""
    msg = "\n".join([method.upper(), path, timestamp, body_hash]).encode()
    expected = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# -----------------------
# API KEY AUTH
# -----------------------
try:
    API_KEYS = json.loads(os.environ["API_KEYS_JSON"])
except KeyError:
    raise RuntimeError("Missing env var: API_KEYS_JSON")
except json.JSONDecodeError:
    raise RuntimeError("API_KEYS_JSON is invalid JSON")


def require_api_key(f):
    endpoint_scope_map = {
        "api_predict": "predict",
        "api_retrieve": "retrieve",
        "api_ingest_kb": "ingest_kb",
        "api_feedback": "feedback",
        "api_health": "health",
    }

    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("x-api-key") or request.args.get("api_key")

        if not isinstance(key, str) or not key or key not in API_KEYS:
            return jsonify({"error": "Unauthorized - missing/invalid API key"}), 401

        endpoint = request.endpoint
        if not endpoint:
            return jsonify({"error": "Invalid endpoint"}), 400

        required_scope = endpoint_scope_map.get(endpoint)
        allowed_scopes = API_KEYS[key]["scopes"]

        if required_scope and required_scope not in allowed_scopes:
            return jsonify({"error": "Forbidden"}), 403

        if not enforce_rate_limit(key, endpoint):
            safe_log_warning("Rate limit exceeded for key suffix: %s on %s", key[-6:], endpoint)
            return jsonify({
                "error": "rate_limit_exceeded",
                "endpoint": endpoint,
                "retry_after_seconds": 60,
            }), 429

        timestamp = request.headers.get("x-timestamp")
        signature = request.headers.get("x-signature")

        if not timestamp or not signature:
            return jsonify({"error": "missing_signature"}), 401

        if not verify_hmac_signature(
            secret=API_KEYS[key]["secret"],
            method=request.method,
            path=request.path,
            timestamp=timestamp,
            body=request.get_data(),
            signature=signature,
        ):
            return jsonify({"error": "invalid_signature"}), 401

        return f(*args, **kwargs)

    return decorated


def get_api_key():
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return {"status": "error", "message": "empty", "status_code": 400}
    if not isinstance(api_key, str):
        return {"status": "error", "message": "API Key must be a string", "status_code": 400}
    return {"status": "success", "key": api_key.strip()}


def get_request_id():
    return str(uuid.uuid4())


def get_safe_bool(val):
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def is_meaningful_text(text):
    if not text:
        return False
    stripped = re.sub(r"\s+", "", text)
    if len(stripped) < 6:
        return False
    if not re.search(r"[a-zA-Z0-9]", stripped):
        return False
    if len(set(stripped)) <= 3:
        return False
    if not re.search(r"[a-zA-Z0-9]{3,}", stripped):
        return False
    return True


def extract_files_from_sources(sources_used):
    files = set()
    for src in sources_used:
        for part in src.split():
            if part.startswith("[FILE:"):
                files.add(part.replace("[FILE:", "").replace("]", ""))
    return sorted(files)


# -----------------------
# RETRIEVAL PIPELINE
# -----------------------
def run_full_retrieval_pipeline(purpose, customer_context_summary, key_points,
                                ops_query, workflow_type, bert_model):
    key_points_list = [kp.strip() for kp in key_points.splitlines() if kp.strip()]
    key_points_with_pipe = ("Key constraints: " + " | ".join(key_points_list)) if key_points_list else ""
    key_points_without_pipe = " ".join(key_points_list) if key_points_list else ""

    cleaned_key_points_with_pipe = clean_key_points(key_points_with_pipe)
    cleaned_key_points_without_pipe = clean_key_points(key_points_without_pipe)
    cleaned_purpose = clean_purpose(purpose.strip())

    if not cleaned_purpose:
        return {"status": "error", "message": "purpose is empty or invalid", "status_code": 400}

    if customer_context_summary:
        intent_results = predict_intent(customer_context_summary["summary"])
        query_for_retrieval = build_retrieval_query(
            customer_context_summary=customer_context_summary,
            key_points=cleaned_key_points_without_pipe,
        )
    elif ops_query:
        intent_results = predict_intent(ops_query)
        query_for_retrieval = build_retrieval_query(
            ops_query=ops_query,
            key_points=cleaned_key_points_without_pipe,
        )
    else:
        return {"status": "error", "message": "input_is_empty", "status_code": 400}

    intent = intent_results["intent"]
    intent_confidence = intent_results["intent_confidence"]
    intent_model_used = intent_results["intent_model_used"]
    intent_status = intent_results["intent_status"]

    context_text, similarity_scores, retrieval_confidence, sources_used, debug_results = retrieve_context(
        bert_model=bert_model,
        query_text=query_for_retrieval,
    )

    interaction_id = store_interaction_state(
        purpose=cleaned_purpose,
        key_points=cleaned_key_points_with_pipe,
        customer_context_summary=customer_context_summary,
        workflow_type=workflow_type,
        ops_query=ops_query,
        intent=intent,
        intent_model_used=intent_model_used,
        intent_confidence=intent_confidence,
        retrieved_context=context_text,
        retrieval_confidence=retrieval_confidence,
        ttl_seconds=INTERACTION_TTL.total_seconds(),
    )

    return {
        "status": "success",
        "context_text": context_text,
        "similarity_scores": similarity_scores,
        "retrieval_confidence": retrieval_confidence,
        "sources_used": sources_used,
        "retrieval_debug": debug_results,
        "customer_context_summary": customer_context_summary,
        "intent": intent,
        "intent_model_used": intent_model_used,
        "intent_confidence": intent_confidence,
        "intent_status": intent_status,
        "interaction_id": interaction_id,
    }

# -----------------------
# GENERATION PIPELINE
# -----------------------
def run_full_generation_pipeline(interaction_id, request_id):
    start_time = time.time()

    state = get_interaction_state(interaction_id=interaction_id)
    if not state:
        return {"status": "error", "message": "Interaction expired. Please retry.", "status_code": 410}

    intent = state["intent"]
    intent_model_used = state["intent_model_used"]
    intent_confidence = state["intent_confidence"]
    purpose = state["purpose"]
    key_points = state["key_points"]
    customer_context_summary = state["customer_context_summary"]
    workflow_type = state["workflow_type"]
    ops_query = state["ops_query"]
    retrieval_confidence = state["retrieval_confidence"]
    retrieved_context = state["retrieved_context"]

    generated_reply = ""

    if customer_context_summary:
        try:
            generated_reply = generate_support_draft(
                purpose=purpose,
                summary=customer_context_summary["summary"],
                retrieved_context=retrieved_context,
                key_points=key_points,
            )
        except ValueError:
            safe_log_error("Support draft generation failed at request ID: %s", request_id)
            return {"status": "error", "message": "LLM generation failed", "status_code": 400}
        except RuntimeError:
            safe_log_error("Support draft generation failed at request ID: %s", request_id)
            return {"status": "error", "message": "LLM generation failed", "status_code": 502}
        except Exception:
            safe_log_error("Support draft generation failed at request ID: %s", request_id)
            return {"status": "error", "message": "Internal server error", "status_code": 500}

    elif ops_query:
        try:
            generated_reply = generate_ops_response(
                purpose=purpose,
                ops_query=ops_query,
                retrieved_context=retrieved_context,
                key_points=key_points,
                workflow_type=workflow_type,
            )
        except ValueError:
            safe_log_error("Ops response generation failed at request ID: %s", request_id)
            return {"status": "error", "message": "LLM generation failed", "status_code": 400}
        except RuntimeError:
            safe_log_error("Ops response generation failed at request ID: %s", request_id)
            return {"status": "error", "message": "LLM generation failed", "status_code": 502}
        except Exception:
            safe_log_error("Ops response generation failed at request ID: %s", request_id)
            return {"status": "error", "message": "Internal server error", "status_code": 500}

    latency_ms = int((time.time() - start_time) * 1000)

    log_api_predict_results(
        tenant_id="default",
        request_id=request_id,
        workflow_type=workflow_type,
        purpose=purpose,
        key_points=key_points,
        customer_context_summary=customer_context_summary,
        ops_query=ops_query,
        intent=intent,
        generated_reply=generated_reply,
        latency_ms=latency_ms,
    )

    return {
        "status": "success",
        "ops_query": ops_query,
        "workflow_type": workflow_type,
        "generated_reply": generated_reply,
        "intent": intent,
        "intent_model_used": intent_model_used,
        "intent_confidence": intent_confidence,
        "purpose": purpose,
        "key_points": key_points,
        "retrieval_confidence": retrieval_confidence,
    }


# -----------------------
# FEEDBACK
# -----------------------
def store_feedback(*, interaction_id, rating, comment=None):
    if rating < 1 or rating > 5:
        return {"status": "error", "message": "rating must be between 1 and 5", "status_code": 400}

    feedback_type = "positive" if rating >= 4 else "neutral" if rating == 3 else "needs_review"

    log = {
        "interaction_id": interaction_id,
        "rating": rating,
        "feedback_type": feedback_type,
        "comment": comment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    log_queue.put((os.path.join(LOG_DIR, "feedback.log"), json.dumps(log)))
    return {"status": "success"}


# -----------------------
# FLASK APP
# -----------------------
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
app.config["DEBUG"] = False
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# -----------------------
# HEALTH
# -----------------------
@app.route("/api/health", methods=["GET"])
@require_api_key
def api_health():
    db_ok = True
    try:
        conn = get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        finally:
            release_pg_conn(conn)
    except Exception:
        safe_log_error("DB health check failed.")
        db_ok = False

    try:
        bert = load_bert_model()
        bert_ok = bert is not None
    except Exception:
        bert_ok = False

    try:
        _ = predict_intent("test")
        intent_ok = True
    except Exception:
        intent_ok = False

    try:
        from llm_client import get_openai_client
        client = get_openai_client()
        gpt_ok = client is not None
    except Exception:
        gpt_ok = False

    overall_ok = db_ok and bert_ok and intent_ok and gpt_ok

    return jsonify({
        "status": "ok" if overall_ok else "degraded",
        "uptime_seconds": int(time.time() - APP_START_TIME),
        "dependencies": {
            "postgres": db_ok,
            "bert": bert_ok,
            "gpt": gpt_ok,
            "intent_model": intent_ok,
        },
    }), 200 if overall_ok else 503


# -----------------------
# RETRIEVAL ENDPOINT
# -----------------------
@app.route("/api/retrieve", methods=["POST"])
@require_api_key
def api_retrieve():
    bert = load_bert_model()
    start_time = time.time()
    ip = get_client_ip()
    ip_hash = hash_ip(ip)
    request_id = get_request_id()
    status = "failure"

    uploaded_files = []
    if request.files:
        uploaded_files = request.files.getlist("uploaded_files")
        raw_payload = request.form.get("payload")
        if not raw_payload:
            return jsonify({"error": "Missing JSON payload"}), 400
        try:
            payload = json.loads(raw_payload)
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400
    else:
        if not request.is_json:
            return jsonify({"error": "JSON required"}), 415
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid JSON body"}), 400

    api_key_eval = get_api_key()
    if api_key_eval["status"] == "error":
        return api_error(api_key_eval["message"], api_key_eval["status_code"])

    api_key = api_key_eval["key"]
    entry = API_KEYS.get(api_key)

    if not entry or entry.get("status") not in ("active", "revoked"):
        return jsonify({"error": "invalid_or_disabled_api_key"}), 403

    try:
        purpose = payload.get("purpose", "")
        if not isinstance(purpose, str) or not purpose.strip() or len(purpose.strip()) < 5:
            return jsonify({"error": "purpose required (min 5 chars)"}), 400

        support_agent_assistance = get_safe_bool(payload.get("support_agent_assistance", False))
        internal_ops = get_safe_bool(payload.get("internal_ops", False))
        financial_ops = get_safe_bool(payload.get("financial_ops", False))
        compliance_validation = get_safe_bool(payload.get("compliance_validation", False))

        if support_agent_assistance:
            workflow_type = "support_agent_assistance"
        elif internal_ops:
            workflow_type = "internal_ops"
        elif financial_ops:
            workflow_type = "financial_ops"
        elif compliance_validation:
            workflow_type = "compliance_validation"
        else:
            return jsonify({"error": "No valid workflow selected"}), 400

        combined_text = ""
        ops_query = ""
        key_points = payload.get("key_points", "")

        if support_agent_assistance:
            past_email = payload.get("past_email", "")
            if not isinstance(past_email, str) or not past_email.strip():
                return jsonify({"error": "past_email required"}), 400

            cleaned_past_email = process_past_email(past_email.strip())

            if not is_meaningful_text(cleaned_past_email) and len(past_email.strip()) > 5:
                return jsonify({
                    "status": "rejected",
                    "reason": "Insufficient information to process request.",
                }), 400

            combined_customer_text = ""
            past_email_len = len(cleaned_past_email.strip())

            if cleaned_past_email.strip():
                combined_customer_text += "CUSTOMER MESSAGE:\n" + cleaned_past_email

            supporting_materials_added = False
            for result in process_attachments(uploaded_files):
                if result["status"] == "error":
                    return api_error(result["message"], 400)
                if result["text"] == "empty":
                    break
                cleaned = process_attachment_text(result["text"])
                if cleaned:
                    if not supporting_materials_added:
                        combined_customer_text += "\n\nATTACHMENTS:\n"
                        supporting_materials_added = True
                    combined_customer_text += cleaned + "\n\n"

            combined_text, too_long, reason, past_email_len, attach_len = is_text_length_long(
                combined_customer_text, past_email_len
            )

            if too_long:
                status = "escalation_triggered"
                log_escalation(
                    tenant_id="default",
                    request_id=request_id,
                    endpoint="retrieve",
                    escalation_reason=f"{reason}, past_email_len={past_email_len}, attach_len={attach_len}",
                    ip_hash=ip_hash,
                )
                return jsonify({"decision": "ESCALATE", "reason": "input_too_long"}), 200

        else:
            ops_query = payload.get("query", "")
            if not isinstance(ops_query, str) or not ops_query.strip() or len(ops_query.strip()) < 5:
                return jsonify({"error": "query required (min 5 chars)"}), 400

            if len(ops_query) > 200:
                status = "escalation_triggered"
                log_escalation(
                    tenant_id="default",
                    request_id=request_id,
                    endpoint="retrieve",
                    escalation_reason=f"ops_query_length={len(ops_query)}",
                    ip_hash=ip_hash,
                )
                return jsonify({"decision": "ESCALATE", "reason": "ops_query_too_long"}), 200

        if not record_ip_abuse_signal(ip_hash, request.endpoint):
            status = "abuse_detected"
            safe_log_warning("Abuse detected: ip_hash=%s endpoint=%s", ip_hash, request.endpoint)
            return jsonify({
                "error": "temporary_blocked",
                "reason": "abuse_detected",
                "retry_after_seconds": 3600,
            }), 429


        ok, reason = enforce_auto_block_precheck(api_key)
        if not ok:
            return api_error(reason, 403)

        ok, reason = record_security_requests(api_key)
        if not ok:
            return api_error(reason, 403)
        

        record_request_usage(tenant_id="default")

        customer_context_summary = {}
        if combined_text:
            record_llm_usage(tenant_id="default")
            try:
                customer_context_summary = generate_summary(combined_text)
            except ValueError:
                safe_log_error("Summary generation failed at request ID: %s", request_id)
                return api_error("LLM generation failed.", 400)
            except RuntimeError:
                safe_log_error("Summary generation failed at request ID: %s", request_id)
                return api_error("LLM generation failed.", 502)
            except Exception:
                safe_log_error("Summary generation failed at request ID: %s", request_id)
                return api_error("Internal server error", 500)

            if not is_meaningful_text(customer_context_summary.get("summary", "")):
                status = "escalation_triggered"
                log_escalation(
                    tenant_id="default",
                    request_id=request_id,
                    endpoint="retrieve",
                    escalation_reason="meaningless_customer_context_summary",
                    ip_hash=ip_hash,
                )
                return jsonify({"decision": "ESCALATE", "reason": "insufficient_information"}), 200

        retrieval_results = run_full_retrieval_pipeline(
            purpose=purpose,
            customer_context_summary=customer_context_summary,
            key_points=key_points,
            ops_query=ops_query,
            workflow_type=workflow_type,
            bert_model=bert,
        )

        if retrieval_results["status"] == "error":
            return api_error(retrieval_results["message"], 400)

        similarity_scores = retrieval_results.get("similarity_scores", []) or []
        retrieval_confidence = retrieval_results.get("retrieval_confidence", 0.0)
        sources_used = retrieval_results.get("sources_used", []) or []
        best_distance = min(similarity_scores) if similarity_scores else None
        intent = retrieval_results.get("intent", "")
        intent_status = retrieval_results.get("intent_status", "")
        intent_model_used = retrieval_results.get("intent_model_used", False)
        intent_confidence = retrieval_results.get("intent_confidence", 0.0)
        customer_context_summary = retrieval_results.get("customer_context_summary")
        interaction_id = retrieval_results.get("interaction_id")

        allow_hard_escalate = entry.get("allow_hard_escalate", False)
        escalate, reason = should_escalate(
            allow_hard_escalate=allow_hard_escalate,
            intent=intent,
            customer_context_summary=customer_context_summary,
            ops_query=ops_query,
            best_distance=best_distance,
            retrieval_confidence=retrieval_confidence,
        )

        if intent_model_used:
            log_intent_model_status(
                tenant_id="default",
                interaction_id=interaction_id,
                intent=intent,
                intent_model_used=intent_model_used,
                intent_confidence=intent_confidence,
                intent_status=intent_status,
            )

        if escalate:
            status = "escalation_triggered"
            log_escalation(
                tenant_id="default",
                request_id=request_id,
                endpoint="retrieve",
                escalation_reason=reason,
                ip_hash=ip_hash,
            )

            SAFE_REASON_MAP = {
                "retrieval_confidence": "low_confidence",
                "best_distance": "low_confidence",
                "keyword": "high_risk_operation",
                "hard_intent": "high_risk_operation",
            }

            display_reason = next(
                (v for k, v in SAFE_REASON_MAP.items() if k in reason), "low_confidence"
            )
            
            return jsonify({"decision": "ESCALATE", "reason": display_reason}), 200
        
        log_debug_retrieval_results(
            tenant_id="default",
            ip_hash=ip_hash,
            interaction_id=interaction_id,
            request_id=request_id,
            retrieval_results=retrieval_results.get("retrieval_debug", []),
        )

        status = "success"
        # interaction_id is returned to the client and sent back with /api/predict.
        # retrieved_context is not returned directly so that clients cannot see or alter it.
        # The context is stored server-side and fetched via interaction_id in /predict.
        return jsonify({
            "decision": "ALLOW",
            "request_id": request_id,
            "interaction_id": interaction_id,
        })

    except Exception:
        status = "failure"
        safe_log_error("Exception in api/retrieve at request ID: %s", request_id)
        return api_error("Internal Server Error", 500)

    finally:
        latency_ms = int((time.time() - start_time) * 1000)
        log_api_event(
            tenant_id="default",
            endpoint="retrieve",
            status=status,
            latency_ms=latency_ms,
            ip_hash=ip_hash,
            request_id=request_id,
            user_agent=request.headers.get("User-Agent"),
        )


# -----------------------
# PREDICTION ENDPOINT
# -----------------------
@app.route("/api/predict", methods=["POST"])
@require_api_key
def api_predict():
    start_time = time.time()
    ip = get_client_ip()
    ip_hash = hash_ip(ip)
    request_id = get_request_id()
    status = "failure"

    if not request.is_json:
        return jsonify({"error": "JSON required"}), 415

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    api_key_eval = get_api_key()
    if api_key_eval["status"] == "error":
        return api_error(api_key_eval["message"], api_key_eval["status_code"])

    api_key = api_key_eval["key"]
    entry = API_KEYS.get(api_key)

    if not entry or entry.get("status") not in ("active", "revoked"):
        return jsonify({"error": "invalid_or_disabled_api_key"}), 403

    try:
        interaction_id = payload.get("interaction_id", "")
        if not interaction_id or not isinstance(interaction_id, str):
            return api_error("interaction_id required", 400)

        interaction_id = "".join(c for c in interaction_id if c.isalnum() or c in ("_", "-"))
        if not interaction_id.strip():
            return api_error("interaction_id cannot be empty", 400)

        if not record_ip_abuse_signal(ip_hash, request.endpoint):
            status = "abuse_detected"
            safe_log_warning("Abuse detected: ip_hash=%s endpoint=%s", ip_hash, request.endpoint)
            return jsonify({
                "error": "temporary_blocked",
                "reason": "abuse_detected",
                "retry_after_seconds": 3600,
            }), 429
        
        ok, reason = enforce_auto_block_precheck(api_key)
        if not ok:
            return api_error(reason, 403)

        ok, reason = record_security_requests(api_key)
        if not ok:
            return api_error(reason, 403)

        record_request_usage(tenant_id="default")
        
        ok, reason = record_security_llm_call(api_key)
    
        if not ok:
            return api_error(reason, 403)
        
        record_llm_usage(tenant_id="default")

        generation_results = run_full_generation_pipeline(
            interaction_id=interaction_id,
            request_id=request_id,
        )

        if generation_results["status"] == "error":
            return api_error(generation_results["message"], generation_results["status_code"])

        generated_reply = generation_results.get("generated_reply", "")
        intent = generation_results.get("intent")
        intent_model_used = generation_results.get("intent_model_used")
        intent_confidence = generation_results.get("intent_confidence")
        retrieval_confidence = generation_results.get("retrieval_confidence")
        ops_query = generation_results.get("ops_query")
        workflow_type = generation_results.get("workflow_type")
        purpose = generation_results.get("purpose")
        key_points = generation_results.get("key_points")

        log_interaction_event(
            tenant_id="default",
            interaction_id=interaction_id,
            session_id=None,
            workflow_type=workflow_type,
            purpose=sanitize_internal_input(purpose, "strict"),
            key_points=sanitize_internal_input(key_points, "strict"),
            ops_query=sanitize_internal_input(ops_query, "strict"),
            intent=intent,
            intent_model_used=intent_model_used,
            intent_confidence=intent_confidence,
            retrieval_confidence=retrieval_confidence,
            decision_outcome="allowed",
            generated_response=generated_reply,
        )

        status = "success"
        return jsonify({
            "decision": "ALLOW",
            "request_id": request_id,
            "generated_reply": generated_reply,
        })

    except Exception:
        status = "failure"
        safe_log_error("Exception in api/predict at request ID: %s", request_id)
        return api_error("Internal Server Error", 500)

    finally:
        latency_ms = int((time.time() - start_time) * 1000)
        log_api_event(
            tenant_id="default",
            endpoint="predict",
            status=status,
            latency_ms=latency_ms,
            ip_hash=ip_hash,
            request_id=request_id,
            user_agent=request.headers.get("User-Agent"),
        )


# -----------------------
# KB INGESTION ENDPOINT
# -----------------------
@app.route("/api/ingest-kb", methods=["POST"])
@require_api_key
def api_ingest_kb():
    bert = load_bert_model()
    start_time = time.time()
    ip = get_client_ip()
    ip_hash = hash_ip(ip)
    request_id = get_request_id()
    status = "failure"

    uploaded_kb = []
    if request.files:
        uploaded_kb = request.files.getlist("uploaded_kb")
        uploaded_kb = [f for f in uploaded_kb if f and f.filename]
        if not uploaded_kb:
            return jsonify({"error": "No valid KB files provided"}), 400
        raw_payload = request.form.get("payload")
        if not raw_payload:
            return jsonify({"error": "Missing JSON payload"}), 400
        try:
            payload = json.loads(raw_payload)
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400
    else:
        if not request.is_json:
            return jsonify({"error": "JSON required"}), 415
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid JSON body"}), 400

    api_key_eval = get_api_key()
    if api_key_eval["status"] == "error":
        return api_error(api_key_eval["message"], api_key_eval["status_code"])

    api_key = api_key_eval["key"]
    entry = API_KEYS.get(api_key)

    if not entry or entry.get("status") not in ("active", "revoked"):
        return jsonify({"error": "invalid_or_disabled_api_key"}), 403

    error_occurred = None
    cleaned_paths = []
    raw_paths = []

    try:
        overwrite = get_safe_bool(payload.get("overwrite_for_uploads", False))
        client_file_hash = payload.get("file_hash", "")

        existing_files = {os.path.basename(p) for p in get_all_kb_files()}

        if not record_ip_abuse_signal(ip_hash, request.endpoint):
            status = "abuse_detected"
            safe_log_warning("Abuse detected: ip_hash=%s endpoint=%s", ip_hash, request.endpoint)
            return jsonify({
                "error": "temporary_blocked",
                "reason": "abuse_detected",
                "retry_after_seconds": 3600,
            }), 429

        ok, reason = enforce_auto_block_precheck(api_key)
        if not ok:
            return api_error(reason, 403)

        ok, reason = record_security_requests(api_key)
        if not ok:
            return api_error(reason, 403)

        record_request_usage(tenant_id="default")
        ingested = []

        try:
            for f in uploaded_kb:
                incoming_name = sanitize_filename(f.filename)

                if incoming_name in existing_files and not overwrite:
                    return api_error("kb_file_already_exists", 400, {
                        "filename": f.filename,
                        "hint": "Set overwrite_for_uploads=true to replace",
                    })

                f.seek(0)
                raw_bytes = f.read()

                ingestion_results = ingest_kb_file(
                    uploaded_file=f,
                    request_id=request_id,
                    safe_filename=incoming_name,
                    raw_bytes=raw_bytes,
                    client_hash=client_file_hash,
                    overwrite=overwrite,
                )

                if ingestion_results["status"] == "error":
                    error_occurred = True
                    if ingestion_results["message"] == "exception":
                        raise Exception("ingestion exception")
                    return api_error(ingestion_results["message"], ingestion_results["status_code"])

                cleaned_paths.append(ingestion_results["cleaned_path"])
                raw_paths.append(ingestion_results["raw_path"])

                result_check = index_uploaded_file(ingestion_results["cleaned_path"], bert)
                if result_check["status"] == "error":
                    return api_error(result_check["message"], 400)

                ingested.append(incoming_name)

        except Exception:
            for path in cleaned_paths:
                try:
                    os.remove(path)
                except Exception:
                    pass
            for path in raw_paths:
                try:
                    os.remove(path)
                except Exception:
                    pass
            try:
                clear_uploads_vector_store()
            except Exception:
                pass
            rebuild_index(bert)
            safe_log_error("KB ingestion failed at request ID: %s", request_id)
            raise

        status = "success"
        return jsonify({"request_id": request_id, "files_ingested": ingested, "status": "ok"})

    except Exception:
        status = "failure"
        safe_log_error("KB ingestion failed at request ID: %s", request_id)
        raise

    finally:
        latency_ms = int((time.time() - start_time) * 1000)
        log_api_event(
            tenant_id="default",
            endpoint="ingest-kb",
            status=status,
            latency_ms=latency_ms,
            ip_hash=ip_hash,
            request_id=request_id,
            user_agent=request.headers.get("User-Agent"),
        )


# -----------------------
# FEEDBACK ENDPOINT
# -----------------------
@app.route("/api/feedback", methods=["POST"])
@require_api_key
def api_feedback():
    start_time = time.time()
    ip = get_client_ip()
    ip_hash = hash_ip(ip)
    request_id = get_request_id()
    status = "failure"

    if not request.is_json:
        return jsonify({"error": "JSON required"}), 415

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    api_key_eval = get_api_key()
    if api_key_eval["status"] == "error":
        return api_error(api_key_eval["message"], api_key_eval["status_code"])

    api_key = api_key_eval["key"]
    entry = API_KEYS.get(api_key)

    if not entry or entry.get("status") not in ("active", "revoked"):
        return jsonify({"error": "invalid_or_disabled_api_key"}), 403

    try:
        if not record_ip_abuse_signal(ip_hash, request.endpoint):
            status = "abuse_detected"
            return jsonify({
                "error": "temporary_blocked",
                "reason": "abuse_detected",
                "retry_after_seconds": 3600,
            }), 429


        ok, reason = enforce_auto_block_precheck(api_key)
        if not ok:
            return api_error(reason, 403)

        ok, reason = record_security_requests(api_key)
        if not ok:
            return api_error(reason, 403)

        record_request_usage(tenant_id="default")

        interaction_id = payload.get("interaction_id")
        rating = payload.get("rating")
        comment = payload.get("comment")

        if not interaction_id or not isinstance(interaction_id, str):
            return api_error("interaction_id required", 400)

        interaction_id = "".join(c for c in interaction_id if c.isalnum() or c in ("_", "-"))
        if not interaction_id.strip():
            return api_error("interaction_id cannot be empty", 400)

        if not rating or not isinstance(rating, int) or not (1 <= rating <= 5):
            return api_error("rating must be an integer between 1 and 5", 400)

        if comment is not None:
            if not isinstance(comment, str):
                return api_error("comment must be a string", 400)
            comment = comment.strip() or None

        result = store_feedback(interaction_id=interaction_id, rating=rating, comment=comment)
        if result["status"] == "error":
            return api_error(result["message"], result["status_code"])

        status = "success"
        return jsonify({"status": "ok", "request_id": request_id}), 200

    except Exception:
        status = "failure"
        safe_log_error("Feedback submission failed at request ID: %s", request_id)
        return api_error("Feedback submission failed", 500)

    finally:
        latency_ms = int((time.time() - start_time) * 1000)
        log_api_event(
            tenant_id="default",
            endpoint="feedback",
            status=status,
            latency_ms=latency_ms,
            ip_hash=ip_hash,
            request_id=request_id,
            user_agent=request.headers.get("User-Agent"),
        )


# -----------------------
# GLOBAL ERROR HANDLERS
# -----------------------
# Stack traces should never be exposed to clients — error handlers return
# clean JSON regardless of what went wrong internally.
@app.errorhandler(404)
def not_found(e):
    return api_error("not_found", 404)

@app.errorhandler(405)
def method_not_allowed(e):
    return api_error("method_not_allowed", 405)

@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.error("Unhandled exception", exc_info=True)
    return jsonify({"error": "internal_server_error"}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False
