# Tajstack AI — Production AI Automation System

A multi-tenant AI automation system with multi-signal escalation guardrails that determines whether a query should be **answered**, **refused**, or **escalated** to a human before a response is generated — designed for high-risk operational workflows where wrong answers are worse than no answer.

Built entirely from scratch by one person over 7 months. Live on AWS EC2.

## Table of Contents
- [The Problem This Solves](#the-problem-this-solves)
- [Architecture](#architecture)
- [Two-Step API Design](#two-step-api-design)
- [Retrieval Confidence Scoring](#retrieval-confidence-scoring)
- [Dual-Index FAISS Retrieval](#dual-index-faiss-retrieval)
- [Intent Classification](#intent-classification)
- [Escalation Logic](#escalation-logic)
- [Security Implementation](#security-implementation)
- [PostgreSQL Design](#postgresql-design)
- [Async Disk Writes](#async-disk-writes)
- [Supported Workflows](#supported-workflows)
- [KB Ingestion Pipeline](#kb-ingestion-pipeline)
- [Deduplication Pipeline](#deduplication-pipeline)
- [Project Structure](#project-structure)
- [Stack](#stack)
- [Running Locally](#running-locally)
- [API Flow Example](#api-flow-example)

---

## The Problem This Solves

AI systems fail silently. When retrieval confidence is low or intent is high-risk, the system still generates — and produces confident wrong answers. In workflows like customer support, finance, or compliance, that is worse than saying nothing.

This system adds a **control layer** between retrieval and generation. Every request is scored before generation runs. If confidence is below threshold, or if intent or keywords signal high risk, the system escalates to a human instead of generating.

**Core principle: better to escalate than be wrong.**

This matters most in fintech, insurtech, legal-tech, and compliance-heavy workflows where a confident wrong answer creates liability.

---

## Architecture

```
Client Request
      │
      ▼
┌─────────────────────┐
│  HMAC-SHA256 Auth   │  Request signing + replay attack protection
│  API Key Validation │  Scope-based access control
│  Rate Limiting      │  Sliding window per key + IP abuse detection
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  /api/retrieve      │
│                     │
│  1. Input cleaning  │  Email thread stripping, attachment parsing 
│  2. Summarization   │  GPT-4o-mini → structured JSON summary + key facts
│  3. Intent classify │  MiniLM embeddings + Logistic Regression (5 classes)
│  4. FAISS retrieval │  Dual-index: knowledge_base/ + uploads/
│  5. Confidence gate │  Custom weighted score (strength + separation + concentration)
│  6. Escalation gate │  Intent + keyword + distance + confidence checks
│                     │
│  → Returns interaction_id only (context stored server-side)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  /api/predict       │
│                     │
│  Fetches state by   │  Retrieved context never sent to client
│  interaction_id     │  Client cannot tamper with retrieval results
│  → LLM generation  │  GPT-4o-mini with strict grounding prompts
└────────┬────────────┘
         │
         ▼
    ALLOW / ESCALATE
```

<img width="850" height="566" alt="image" src="https://github.com/user-attachments/assets/69ca8bd8-04f8-412d-a9ce-4d755e4fd8c0" />

---

## Two-Step API Design

Most RAG systems return retrieved context to the client, who then passes it back for generation. This creates a prompt injection risk — a client could alter the retrieved context before the generation step.

This system stores retrieved context server-side in PostgreSQL after `/api/retrieve`, identified by a UUID `interaction_id`. The client gets only the ID. `/api/predict` fetches the context by ID and generates from it. The client cannot see or modify the retrieved context at any point.

---

## Retrieval Confidence Scoring

Standard RAG systems use raw distance scores from vector search. This system computes a custom confidence metric from three signals:

```python
strength      = 1.0 - (top1 / 0.8)          # how good is the best match in absolute terms?
separation    = gap / 0.8                    # how much better is rank-1 vs rank-2?
concentration = (avg_rest - top1) / 0.8     # how much better is rank-1 vs the average of the rest

confidence = 0.5 * strength + 0.3 * separation + 0.2 * concentration
```

If confidence falls below threshold, the system escalates regardless of what the LLM would have said.

---

## Dual-Index FAISS Retrieval

Two separate FAISS indexes are maintained:

- `knowledge_base/` — permanent KB files placed manually, indexed once at first request and loaded from disk on subsequent requests
- `uploads/` — files ingested via `/api/ingest-kb`, indexed incrementally without rebuilding the full index

Each directory has its own FAISS index. They are searched separately with different `top_k` budgets (5 for KB, 3 for uploads), then merged and re-ranked by normalized L2 distance.

---

## Intent Classification

A lightweight MiniLM + Logistic Regression classifier trained on a custom labeled dataset with 5 intent classes:

```
refund_request | order_status | technical_issue | complaint | general_query
```

Why not fine-tune a full transformer:
- MiniLM embeddings are high quality and generalize well on this task
- Logistic Regression inference is microseconds vs. transformer forward pass
- Interpretable — probabilities are meaningful confidence scores
- The same MiniLM model is already loaded for retrieval embeddings — no extra memory cost

Intent is used in the escalation gate: `refund_request` and `complaint` trigger hard escalation for high-risk workflow configurations, regardless of retrieval confidence.

---

## Escalation Logic

Escalation runs before generation. Conditions checked in order:

1. **Hard intent escalation** — `refund_request`, `complaint` for high-risk clients
2. **Retrieval confidence below threshold** — system is uncertain, escalate
3. **Best distance exceeds threshold** — top result is too far from query
4. **Keyword detection** — legal, lawyer, lawsuit, gdpr, chargeback, etc.

None of these are configurable at request time — they are system-level constants. Clients cannot disable escalation by passing parameters.

---

## Security Implementation

**HMAC-SHA256 Request Signing**

Every request is signed over `method + path + timestamp + sha256(body)`. The server recomputes the signature using the stored secret and compares with `hmac.compare_digest` (constant-time comparison, prevents timing attacks). A 5-minute timestamp window prevents replay attacks.

**IP Hashing**

Client IPs are never stored raw. Before logging, each IP is hashed with SHA-256 + a server-side salt:
```python
hashlib.sha256(f"{IP_HASH_SALT}:{ip}".encode()).hexdigest()
```
This makes logs safe to retain without exposing client infrastructure.

**Rate Limiting + Abuse Detection**

Two independent layers:
- Per-API-key sliding window rate limiter (60-second window, per endpoint)
- Per-IP hourly abuse detector (independent from key rotation — an attacker cycling keys is still caught)

**Auto-Block**

Monthly caps on requests and LLM calls per API key. If exceeded — whether from a bug, infinite loop, or genuine abuse — the key is auto-disabled.

**File Security**

KB files are written to a secure temp directory (`mode=0o700`) during validation, then moved to permanent storage. Raw files are stored read-only (`chmod 0o400`) as immutable audit records. Cleaned text is written atomically via `NamedTemporaryFile` + `os.replace()` — crash-safe.

---

## PostgreSQL Design

- **Serializable isolation level** on all writes — prevents lost updates under concurrent workers without application-level locking
- **Connection pooling** via `psycopg2.SimpleConnectionPool` — reuses connections instead of opening a new TCP connection per request
- **Retry loop with exponential backoff** on `SerializationFailure` — handles contention gracefully under load
- **`ON CONFLICT DO UPDATE`** for usage tracking — single upsert instead of separate read-modify-write, safe under concurrency

---

## Async Disk Writes

Log writes are queued to a background thread via `queue.Queue`. This removes file I/O from the request critical path entirely — a request handling 4-6 log files per call would otherwise block on disk latency under concurrent load.

```python
log_queue = queue.Queue()

def log_writer():
    while True:
        path, data = log_queue.get()
        with open(path, "a") as f:
            f.write(data + "\n")
        log_queue.task_done()

threading.Thread(target=log_writer, daemon=True).start()
```

---

## Supported Workflows

| Workflow | Description | Generation Style |
|---|---|---|
| `support_agent_assistance` | Drafts customer-facing email replies for support agents | Empathetic, policy-grounded |
| `internal_ops` | Answers internal operational queries from KB | Factual, no decisions |
| `financial_ops` | Advisory responses for finance/revenue queries | No approvals, no commitments |
| `compliance_validation` | Presents what policy says, never interprets | Neutral, no compliance conclusions |

---

## KB Ingestion Pipeline

`POST /api/ingest-kb` accepts `.txt`, `.pdf`, `.json` files:

1. SHA-256 hash verification (optional client-side hash matching)
2. Secure temp directory write (`mode=0o700`)
3. Text extraction (`pdfplumber` for PDF, semantic conversion for JSON)
4. Boilerplate removal + content validation
5. Atomic write to `uploads/` (`NamedTemporaryFile` + `os.replace`)
6. Raw file stored read-only as audit record
7. Incremental FAISS indexing — existing index extended, not rebuilt
8. On any failure: cleanup cleaned files + raw files + clear vector store + rebuild from remaining good files

---

## Deduplication Pipeline

Before indexing, chunks go through three deduplication passes:

1. **Exact text dedupe** — removes chunks with identical content
2. **Substring dedupe** — removes chunk A if its full text appears inside chunk B
3. **Semantic dedupe** — removes near-duplicates above cosine similarity threshold (0.96), with source priority: `UPLOADED_KB > KB > URL`

This keeps the vector space clean and prevents a single repeated fact from dominating retrieval results.

---

## Project Structure

```
├── app.py                      # Flask API layer — all endpoints
├── main_logic.py               # Core logic — retrieval, FAISS, DB, security
├── intent_model/               # Saved MiniLM embedder (generated by intent_train.py)
│   ├── intent_train.py/        # Intent classifier training pipeline
│   └── intent_dataset.csv/     # Intent classification dataset
├── knowledge_base/             # Permanent KB files (.txt, .pdf, .json)
├── uploads/                    # KB files ingested via /api/ingest-kb
├── vector_store/
│   ├── kb/                     # FAISS index + metadata for knowledge_base/
│   └── uploads/                # FAISS index + metadata for uploads/
├── logs/                       # Structured JSON logs per event type
├── usage/                      # Monthly usage tracking (JSON)
├── security/                   # Per-key security state + auto-block caps
└── raw_files/                  # Immutable audit copies of ingested KB files

```

---

## Stack

| Component | Technology |
|---|---|
| API framework | Flask + Gunicorn |
| Vector search | FAISS (IndexFlatL2) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Intent classifier | MiniLM + Logistic Regression (scikit-learn) |
| Sentiment analysis | DistilBERT (HuggingFace pipeline) |
| LLM | OpenAI GPT-4o-mini |
| Database | PostgreSQL (psycopg2) |
| Infrastructure | AWS EC2, S3, CloudWatch, CloudTrail |
| Auth | HMAC-SHA256 request signing + API key scopes |

---

## Environment Variables Required

```
OPENAI_API_KEY        # OpenAI API key
PG_DSN                # PostgreSQL connection string
API_KEYS_JSON         # JSON object mapping API keys to scopes and secrets
IP_HASH_SALT          # Salt for IP hashing before log storage
```

---

## Running Locally

> **Note:** Retrieval runs entirely locally without an OpenAI key. Only `/api/predict`, `/api/retrieve` and other API endpoints require `OPENAI_API_KEY`.

```bash
pip install -r requirements.txt

# Train intent classifier first
python intent_train.py

# Place KB files in knowledge_base/
# A sample KB file is provided in knowledge_base/sample_policy.txt

# Verify retrieval is working without an OpenAI key
python retrieval_test.py

# Set environment variables, then run the full server
python app.py
```

---

## API Flow Example

```bash
# Step 1 — Retrieve (returns interaction_id)
POST /api/retrieve
{
  "purpose": "Handle customer refund query",
  "support_agent_assistance": true,
  "past_email": "I need a refund for order #12345..."
}

# Response
{
  "decision": "ALLOW",
  "interaction_id": "uuid-here",
  "request_id": "uuid-here"
}

# Step 2 — Generate (uses stored context, never re-sent by client)
POST /api/predict
{
  "interaction_id": "uuid-here"
}

# Response
{
  "decision": "ALLOW",
  "generated_reply": "Thank you for reaching out..."
}

# If escalation triggers at Step 1
{
  "decision": "ESCALATE",
  "reason": "low_confidence"
}
```
