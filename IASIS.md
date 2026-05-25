# IASIS AI — Medical Conversational AI Server Specification

## PROJECT GOAL

Build a deployable AI-powered medical assistant backend server that:

1. Chats with patients conversationally
2. Asks follow-up medical questions dynamically
3. Predicts possible diseases from symptoms
4. Accepts medical reports (PDF/images)
5. Extracts and analyzes report data
6. Combines:
   - symptom analysis
   - report analysis
   - patient answers
7. Returns:
   - possible diseases
   - urgency level
   - recommendations
   - follow-up questions
8. Runs on a VPS
9. Exposes REST APIs for frontend communication

This is NOT a hospital-grade diagnostic system.
All responses must include medical disclaimers.

---

# SYSTEM ARCHITECTURE

```text
Frontend (Next.js)
        ↓ REST API
FastAPI AI Server
        ↓
────────────────────────────
1. Conversation Engine
2. Symptom Extraction Engine
3. Disease Prediction Model
4. Follow-Up Question Engine
5. Report Parser (PDF/OCR)
6. Medical Prompt Builder
7. LLM Inference Layer
8. Emergency Detection Engine
────────────────────────────
        ↓
JSON Response
```

---

# CORE TECHNOLOGY STACK

## Backend

- Python 3.11+
- FastAPI
- Uvicorn

## ML / AI

- scikit-learn
- XGBoost
- joblib
- Ollama
- Llama 3 / Qwen2.5 / Gemma

## NLP

- spaCy
- regex symptom extraction

## Report Parsing

- pdfplumber
- pytesseract
- Pillow

## Deployment

- Docker
- DigitalOcean VPS

---

# PROJECT STRUCTURE

```text
iasis-ai-server/
│
├── app/
│   ├── main.py
│   │
│   ├── routes/
│   │   ├── chat.py
│   │   ├── analyze.py
│   │   ├── report.py
│   │   └── health.py
│   │
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── predictor_service.py
│   │   ├── prompt_service.py
│   │   ├── symptom_extractor.py
│   │   ├── report_parser.py
│   │   ├── ocr_service.py
│   │   ├── followup_engine.py
│   │   ├── emergency_engine.py
│   │   └── medical_rules.py
│   │
│   ├── models/
│   │   ├── disease_model.pkl
│   │   ├── symptom_encoder.pkl
│   │   └── schemas.py
│   │
│   ├── prompts/
│   │   ├── triage_prompt.txt
│   │   ├── report_prompt.txt
│   │   └── followup_prompt.txt
│   │
│   └── utils/
│
├── datasets/
├── uploads/
├── notebooks/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# PRIMARY FEATURES

## FEATURE 1 — Conversational Medical Chat

### Requirements

- User sends symptom messages
- AI responds conversationally
- AI asks follow-up medical questions
- Support English + Bangla mixed input

### Example

User:

```text
I have fever and chest pain
```

AI:

```text
How long have you had chest pain?
Do you also have shortness of breath?
```

---

## FEATURE 2 — Disease Prediction Engine

### Goal

Train a symptom-to-disease prediction model.

### Model Type

Use:

- XGBoost
- RandomForest
- LightGBM

DO NOT use heavy neural networks initially.

### Training Dataset Format

```csv
fever,cough,chest_pain,headache,disease
1,1,1,0,pneumonia
1,0,0,1,dengue
```

### Training Pipeline

1. Load dataset
2. Encode symptoms
3. Train classifier
4. Evaluate accuracy
5. Save model using joblib

### Required Training Script

File:

```text
train_model.py
```

Responsibilities:

- preprocessing
- train/test split
- training
- metrics
- save model

---

## FEATURE 3 — Dynamic Follow-Up Question Engine

### Goal

Ask intelligent follow-up questions.

### Example Rules

```python
if "chest pain" detected:
    ask about:
    - shortness of breath
    - sweating
    - duration
```

Questions should depend on:
- predicted diseases
- symptoms
- urgency

---

## FEATURE 4 — Medical Report Analysis

### Supported Inputs

- PDF
- JPG
- PNG

### PDF Extraction

Use:
- pdfplumber

### OCR Extraction

Use:
- pytesseract

### Output

Extract:
- blood values
- abnormal values
- report summaries

---

## FEATURE 5 — LLM Layer

### Use Ollama

Install:

```bash
ollama pull llama3
```

OR:

```bash
ollama pull qwen2.5
```

### LLM Responsibilities

- conversational responses
- report explanation
- recommendation generation
- urgency reasoning

### IMPORTANT RULE

LLM MUST NOT directly diagnose.

Allowed wording:
- possible condition
- may indicate
- consult doctor
- seek emergency care

Forbidden wording:
- you definitely have X

---

## FEATURE 6 — Emergency Detection Engine

### Hardcoded Rule System

Do NOT rely only on AI.

Example:

```python
IF:
- chest pain
- sweating
- shortness of breath

THEN:
urgency = EMERGENCY
```

---

## FEATURE 7 — REST API

### Endpoint 1 — Chat

```http
POST /chat
```

Request:

```json
{
  "message": "I have fever and chest pain",
  "conversation_id": "abc123"
}
```

Response:

```json
{
  "reply": "How long have you had chest pain?",
  "possible_diseases": [
    {
      "name": "Pneumonia",
      "probability": 0.81
    }
  ],
  "urgency": "HIGH"
}
```

### Endpoint 2 — Analyze Report

```http
POST /analyze-report
```

Multipart upload:
- PDF/image
- optional symptom text

### Endpoint 3 — Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## FEATURE 8 — JSON OUTPUT ENFORCEMENT

LLM responses must ALWAYS return valid JSON.

Required structure:

```json
{
  "possible_diseases": [],
  "urgency": "",
  "followup_questions": [],
  "advice": "",
  "disclaimer": ""
}
```

---

## FEATURE 9 — Conversation Memory

Maintain:
- previous symptoms
- previous answers
- report history

Use:
- Redis OR in-memory session cache

---

## FEATURE 10 — Bangla + English Support

Must support mixed language input:

Example:

```text
আমার chest pain হচ্ছে
```

---

# VPS DEPLOYMENT

## Target VPS

Use:
- DigitalOcean

Minimum:
- 4 vCPU
- 8GB RAM

### Deployment Requirements

Provide:
- Dockerfile
- docker-compose.yml

### Reverse Proxy

Use:
- Nginx

### Production Server

Use:

```bash
gunicorn + uvicorn workers
```

---

# SECURITY REQUIREMENTS

- validate uploads
- max 20MB uploads
- sanitize filenames
- CORS configuration
- API rate limiting
- input validation via Pydantic

---

# MODEL REQUIREMENTS

## Disease Prediction Model

Use:
- multi-class classification

Return:
- top 3 diseases
- probability scores

### MODEL FILES

```text
/models/
    disease_model.pkl
    symptom_encoder.pkl
```

---

# DATASETS

Recommended public datasets:

- Symptom2Disease
- Disease Symptom Prediction Dataset
- MedQuAD

---

# REQUIRED DELIVERABLES

The AI agent must generate:

1. Full FastAPI backend
2. Training pipeline
3. Disease prediction model
4. Ollama integration
5. PDF parsing
6. OCR parsing
7. Follow-up question engine
8. Emergency rules engine
9. Docker deployment
10. REST APIs
11. README with setup instructions
12. Example environment variables
13. API documentation
14. Requirements.txt
15. Pydantic schemas

---

# IMPORTANT ENGINEERING RULES

## Use Modular Architecture

Avoid giant files.

## Use Service Layer Pattern

Business logic must stay outside route handlers.

## Use Typed Schemas

Use Pydantic everywhere.

---

# RESPONSE STYLE REQUIREMENTS

All AI responses must:
- be concise
- medically cautious
- include disclaimer
- avoid hallucinations
- avoid guaranteed diagnosis

---

# DISCLAIMER REQUIREMENT

Every medical response must include:

```text
This is AI-generated guidance and not a medical diagnosis.
Consult a licensed doctor for professional medical advice.
```

---

# FUTURE UPGRADE SUPPORT

Architecture must support future:
- RAG
- vector database
- fine-tuning
- multilingual expansion
- doctor review system
- hospital integrations

---

# FINAL GOAL

Build a production-style AI medical assistant backend suitable for:
- university project
- MVP startup
- VPS deployment
- frontend API integration

with modular scalable architecture and clean engineering practices.
