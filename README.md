# IASIS AI — Conversational Medical Triage & Report Analyzer

IASIS AI is a modern, responsive, healthcare-grade medical assistant that combines a **FastAPI backend** with a **React + TypeScript + Vite frontend**. It features cloud LLM-powered symptom triage via the **Groq API** (Qwen3-32B / GPT-OSS-120B), a machine learning disease predictor, automated high-risk emergency detection, and medical document/lab report parsing (using Tesseract OCR and PDF parsers).

---

## 📖 Table of Contents
1. [Key Features](#-key-features)
2. [Architecture Overview](#-architecture-overview)
3. [Project Directory Structure](#-project-directory-structure)
4. [Prerequisites](#-prerequisites)
5. [Backend Installation & Run Guide](#-backend-installation--run-guide)
6. [Training the Machine Learning Predictor](#-training-the-machine-learning-predictor)
7. [Frontend Installation & Run Guide](#-frontend-installation--run-guide)
8. [API Endpoints](#-api-endpoints)
9. [OCR & Document Upload Config](#-ocr--document-upload-config)
10. [Groq LLM Configuration](#-groq-llm-configuration)
11. [Security & Rate Limiting](#-security--rate-limiting)
12. [Medical Disclaimer](#%EF%B8%8F-medical-disclaimer)

---

## 🌟 Key Features

### 💻 Modern Medical Frontend
- **Glassmorphic Medical Theme:** Clean, premium aesthetic with healthcare-curated palettes (soothing greens, deep blues, slate grays).
- **Conversational Chat Interface:** Includes real-time chat, streaming status indicators, and clear turn counters.
- **Urgency Indicator Badge:** Visual alert displays (NONE, LOW, MEDIUM, HIGH, EMERGENCY) based on clinical signs.
- **Interactive Follow-Up Chips:** Recommended follow-up questions render as clickable badges to streamline user responses.
- **Disease Concern Cards:** Interactive panels displaying ML predictions with calibrated concern levels ("High Concern", "Moderate Concern", "Must Rule Out Urgently").
- **Multimodal Document Upload:** Seamlessly upload blood reports, prescriptions, or clinical notes (PDF or Image) directly into the chat flow to receive instant, context-aware analysis that merges directly into the patient's active memory.

### 🧠 V3 Adaptive Conversational Triage
- **Symptom Accumulation Memory:** Preserves symptom history across a `conversation_id`.
- **Intelligent Deduplication:** Merges overlapping symptoms. For instance, if a user reports a "cough" and later specifies it's a "dry cough," the memory upgrades the record to `dry_cough` and preserves duration and severity.
- **Turn-based State Machine:** Tracks asked versus answered follow-up topics to prevent repetitive AI interrogation.
- **Urgency Escalation Lock:** Urgency levels can only escalate within a session to ensure patient safety is never minimized.

### 📊 Hybrid Intelligence Pipeline
1. **Rule-Based Red Flag Engine:** Deterministically bypasses the AI to immediately flag high-risk symptoms (e.g., severe chest pain, stroke symptoms, anaphylaxis) and invokes a `CRITICAL EMERGENCY OVERRIDE` to enforce life-saving directives.
2. **Machine Learning Classifier:** A RandomForest model trained on **773 distinct diseases** predicting qualitative severity concerns, ensuring no hallucinatory LLM percentages are displayed.
3. **Cloud LLM Layer (Groq API):** Uses 32B–120B parameter models (`qwen/qwen3-32b` primary, `openai/gpt-oss-120b` fallback) for chain-of-thought clinical reasoning, adaptive medical interviewing, and structured JSON output.
4. **Report Analyzer:** Tesseract OCR (for images) and `pdfplumber` (for PDFs) extract lab records, which are then analyzed by the LLM and merged into the active conversation state.
5. **Adaptive Test Engine:** Automatically recommends the next best diagnostic tests based on evidence-gaps and explicitly avoids repeating uploaded lab panels, optimizing the diagnostic pathway.

---

## 🏗️ Architecture Overview

```text
               ┌────────────────────────┐
               │    React + Vite Client  │
               │   (Port 5173 / Browser) │
               └───────────┬────────────┘
                           │
                           │ REST APIs (HTTP / JSON / Multipart)
                           ▼
               ┌────────────────────────┐
               │     FastAPI Backend    │
               │      (Port 8000)       │
               └───────────┬────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
 ┌──────────────────┐                  ┌──────────────┐
 │   Groq API (LLM) │                  │ RandomForest │
 │ ┌──────────────┐ │                  │ Classifier   │
 │ │ PRIMARY:     │ │                  └──────────────┘
 │ │ qwen3-32b    │ │                         ▲
 │ ├──────────────┤ │                         │ ML Inference
 │ │ FALLBACK:    │ │                  ┌──────┴──────────────┐
 │ │ gpt-oss-120b │ │                  │ base_symptoms list  │
 │ └──────────────┘ │                  │ (Matches 377 features)
 └──────────────────┘                  └─────────────────────┘
        ▲
        │ Chain-of-Thought Reasoning
 ┌──────┴──────────────┐
 │ Memory State (V3)   │
 │ - asked/answered    │
 │ - accumulated_symps │
 │ - conversation_hist │
 └─────────────────────┘
```

---

## 📂 Project Directory Structure

```text
project-health_care/
├── app/                        # FastAPI Backend Application
│   ├── main.py                 # Core app entry with CORS & Rate Limiting
│   ├── routes/                 # Endpoint routers
│   │   ├── chat.py             # /chat (Symptom processing & state machine)
│   │   ├── analyze.py          # /analyze-report (Multipart file analysis)
│   │   ├── report.py           # /report/* (Upload & fetch document text)
│   │   ├── session.py          # /session/* (Manage & clear memory states)
│   │   └── health.py           # /health (Liveliness checks)
│   ├── services/               # Core business logic services
│   │   ├── memory_service.py   # V3 Conversational state memory
│   │   ├── followup_engine.py  # Follow-up logic and deduplication
│   │   ├── ocr_service.py      # Tesseract image text extraction
│   │   ├── report_parser.py    # pdfplumber PDF text extraction
│   │   ├── llm_service.py      # Groq API inference (primary/fallback cascade)
│   │   ├── system_prompts.py   # Chain-of-thought system prompts for Groq
│   │   ├── predictor_service.py# RandomForest disease predictor
│   │   ├── prompt_service.py   # User prompt template loader
│   │   ├── symptom_extractor.py# NLP symptom token matching
│   │   ├── emergency_engine.py # Rule-based emergency trigger check
│   │   └── medical_rules.py    # List of high-risk terms and question rules
│   ├── models/
│   │   └── schemas.py          # Pydantic schemas (Request/Response & State models)
│   └── prompts/                # Chat, report analysis, & follow-up prompts
│
├── frontend/                   # React + TypeScript + Vite Frontend
│   ├── package.json            # Node project configuration
│   ├── vite.config.ts          # Vite configuration (routes proxy/build targets)
│   ├── src/
│   │   ├── main.tsx            # React root injection
│   │   ├── App.tsx             # Page layout & Router shell
│   │   ├── components/         # Reusable UI Widgets
│   │   │   ├── ChatMessage.tsx # Chat bubble component (User vs Assistant)
│   │   │   ├── DiseaseCard.tsx # Disease probability progress bars
│   │   │   ├── FollowUpChips.tsx# Quick-reply question chips
│   │   │   ├── Sidebar.tsx     # Navigation sidebar
│   │   │   └── UrgencyBadge.tsx# Urgency classification pill
│   │   ├── pages/              # Main view screens
│   │   │   ├── ChatPage.tsx    # Live Chat panel with side panels
│   │   │   └── ReportPage.tsx  # Document analyzer and results view
│   │   ├── services/
│   │   │   └── api.ts          # Frontend API client
│   │   └── types/
│   │       └── api.ts          # Shared TypeScript interfaces
│
├── datasets/                   # CSV files for training models
├── models/                     # Saved model files (disease_model.pkl, encoders)
├── uploads/                    # Directory for temporary file uploads
├── train_model.py              # Script to build and train the RandomForest model
├── generate_dataset.py         # Script to create a synthetic dataset (fallback)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configurations
├── docker-compose.yml          # Docker-compose configuration
├── .env.example                # Template for configurations
└── README.md                   # This project manual
```

---

## 🛠️ Prerequisites

Before getting started, make sure you have the following:
- **Python 3.9 to 3.11**
- **Node.js 18+** & **npm**
- **Groq API Key** — Sign up at [console.groq.com](https://console.groq.com) to get a free API key
- **Tesseract OCR** (for image text recognition)

---

## 🐍 Backend Installation & Run Guide

### 1. Set Up Python Virtual Environment
Navigate to the root directory of the project and create a virtual environment:
```bash
python -m venv venv
```
Activate the environment:
- **Windows (CMD/PowerShell):**
  ```powershell
  venv\Scripts\activate
  ```
- **Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Configuration
Copy the sample environment file to create a local `.env` configuration:
```bash
cp .env.example .env
```
**Important:** Set your `GROQ_API_KEY` in the `.env` file. You can also customize the models:
- `GROQ_PRIMARY_MODEL` — Fast primary model (default: `qwen/qwen3-32b`)
- `GROQ_FALLBACK_MODEL` — Stronger fallback model (default: `openai/gpt-oss-120b`)
- `GROQ_TIMEOUT_SECONDS` — Request timeout (default: `30`)

### 4. Run the FastAPI Server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Once started, the interactive API documentation will be available at:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📊 Training the Machine Learning Predictor

The disease predictor depends on trained pipeline outputs saved to the `models/` directory.

### Option A: Standard Training (Using Kaggle Dataset - Recommended)
1. Download the [Diseases and Symptoms Dataset from Kaggle](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset).
2. Save the CSV file as `datasets/Final_Augmented_dataset_Diseases_and_Symptoms.csv`.
3. Train the model:
   ```bash
   python train_model.py
   ```

### Option B: Quick Mock Training (Synthetic Dataset)
If you just want to test without downloading large external files, generate a synthetic set first:
```bash
python generate_dataset.py
python train_model.py
```
This builds and saves the model objects: `models/disease_model.pkl`, `models/symptom_encoder.pkl`, and `models/symptom_list.json`.

---

## ⚛️ Frontend Installation & Run Guide

### 1. Navigate and Install Packages
Navigate to the frontend directory and install dependencies:
```bash
cd frontend
npm install
```

### 2. Run the Development Server
```bash
npm run dev
```
By default, the React UI will be served at: **[http://localhost:5173](http://localhost:5173)**.

---

## 🔌 API Endpoints

### 🩺 Health & Diagnostics
- `GET /health`: Check server and system status.
- `GET /session/stats`: Return current session storage statistics (active sessions, TTL values).

### 💬 Conversational Triage
- `POST /chat`: Submit a chat turn. Maintains state automatically.
  - **Payload:**
    ```json
    {
      "message": "I'm having a severe headache and nausea",
      "conversation_id": "session-12345",
      "age": 30,
      "gender": "male",
      "chronic_conditions": "hypertension"
    }
    ```
- `DELETE /session/{conversation_id}`: Clear session state, reset symptom list, and free memory for a specific ID.

### 📂 Report Upload & Processing
- `POST /analyze-report`: Accept multipart file uploads (`file`) and optional patient context (`symptoms`). Analyzes, extracts text (PDF or OCR), finds abnormal parameters, and returns an LLM evaluation.
- `POST /report/upload`: Extract plain text from an uploaded medical document and returns a unique `report_id`.
- `GET /report/{report_id}`: Retrieve previously uploaded document text.

---

## 📷 OCR & Document Upload Config

To extract text from images (PNG, JPG), the server utilizes Tesseract OCR.

### Windows Configuration
1. Install Tesseract OCR via the [installer](https://github.com/UB-Mannheim/tesseract/wiki).
2. Ensure Tesseract is installed at the default path `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. If installed in a custom directory, update `pytesseract.pytesseract.tesseract_cmd` inside [ocr_service.py](file:///d:/project-health_care/app/services/ocr_service.py) or define it in your `.env` file under `TESSERACT_CMD`.

### Linux/macOS Configuration
Install using the system package manager:
- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
- **macOS:** `brew install tesseract`

---

## 🚀 Groq LLM Configuration

IASIS AI uses the [Groq API](https://groq.com/) for high-performance cloud LLM inference.

### Model Architecture

The system uses a **dual-model cascade** for reliability and cost optimization:

| Model | Role | Speed | Reasoning | Cost |
|---|---|---|---|---|
| `qwen/qwen3-32b` | **Primary** — handles all requests first | ⚡ ~1-3s | Strong | Low |
| `openai/gpt-oss-120b` | **Fallback** — activates only on primary failure | Moderate ~3-8s | Excellent | Higher |

### Setup

1. Sign up at [console.groq.com](https://console.groq.com) and create an API key.
2. Set your key in `.env`:
   ```bash
   GROQ_API_KEY=gsk_your_api_key_here
   ```
3. (Optional) Customize models and timeout:
   ```bash
   GROQ_PRIMARY_MODEL=qwen/qwen3-32b
   GROQ_FALLBACK_MODEL=openai/gpt-oss-120b
   GROQ_TIMEOUT_SECONDS=30
   GROQ_MAX_RETRIES=2
   ```

### Error Handling

The LLM service implements a production-ready error handling cascade:
1. **Attempt 1** — Primary model (`qwen3-32b`)
2. **Attempt 2** — Fallback model (`gpt-oss-120b`) with exponential backoff
3. **Final fallback** — Hardcoded safe response (no LLM dependency)

Additional protections:
- **Rate limit handling:** Automatically backs off on `429` responses using `Retry-After` headers.
- **JSON repair:** Recovers from common LLM JSON formatting issues (markdown fences, trailing commas).
- **Graceful degradation:** If `GROQ_API_KEY` is missing, the server still starts and returns safe fallback responses.

### Rate Limits (Free Tier)

| Limit | Value |
|---|---|
| Requests/minute | 30 |
| Requests/day | 14,400 |
| Tokens/minute | 6,000 |

For production workloads, consider upgrading to a [paid Groq plan](https://console.groq.com).

---

## 🛡️ Security & Rate Limiting

- **In-Memory Rate Limiting:** Enforces a limit of **30 requests per minute per IP address** to prevent abuse.
- **Upload Restrictions:** Imposes a strict file upload limit of **20MB** and checks MIME types, accepting only `application/pdf`, `image/png`, and `image/jpeg`.
- **Sanitized Data Handling:** Automatically sanitizes filenames before disk storage to prevent path traversal risks.

---

## ⚠️ Medical Disclaimer

> [!WARNING]
> This application is built as an educational tool for conversational medical triage guidance. Under no circumstances should its outputs be treated as official professional medical diagnoses or treatment recommendations. Always advise patients to consult a licensed healthcare practitioner or seek emergency services for serious health concerns.
