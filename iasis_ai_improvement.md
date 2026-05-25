# IASIS AI — Improvement Tasks for Antigravity

## Project Context

The IASIS AI backend architecture is already functional and matches the original IASIS specification.

Current system includes:
- FastAPI backend
- RandomForest disease prediction model
- Ollama LLM integration
- OCR/PDF report parsing
- Emergency detection engine
- Symptom extraction engine
- Follow-up question engine
- Conversation memory
- Docker deployment

The objective now is NOT to rewrite the project.
The objective is to improve:
- medical reasoning quality
- conversational continuity
- extraction quality
- backend robustness
- safety
- production readiness

IMPORTANT:
Do NOT refactor the whole project.
Do NOT rename existing APIs.
Do NOT break frontend compatibility.
Preserve existing architecture.

---

# TASK 1 — Improve Symptom Matching Safety

## Problem

Current symptom matching uses partial matching logic:

```python
if normalized in feat or feat in normalized
```

This can create dangerous false matches.

Examples:
- "pain" may match many unrelated symptoms
- "cold" may match unintended features

## Required Changes

File:

```text
app/services/predictor_service.py
```

Replace partial fuzzy matching with:
- exact normalized matching
- underscore-normalized matching only

## Required Logic

```python
feat_readable = feat.replace("_", " ")

if normalized == feat:
    input_vector[feat] = 1

elif normalized == feat_readable:
    input_vector[feat] = 1
```

## Constraints

- Do NOT change ML model structure
- Do NOT retrain classifier
- Preserve API response format

---

# TASK 2 — Expand Conversation Memory

## Goal

Improve conversational continuity.

Current memory system is too minimal.

## Create

```text
app/services/memory_service.py
```

## Store

For each conversation_id store:

```python
{
  "symptoms": [],
  "predictions": [],
  "urgency": "",
  "followup_questions": [],
  "reports": [],
  "history": []
}
```

## Modify

```text
app/routes/chat.py
```

## Requirements

1. Load memory at start of request
2. Inject memory into prompt builder
3. Update memory after every response
4. Preserve existing API structure
5. Keep service-layer architecture

## Constraints

- Use lightweight in-memory cache only
- Do NOT add Redis yet
- Preserve current endpoints

---

# TASK 3 — Add Severity Extraction

## Goal

Extract symptom severity from user messages.

Examples:
- mild fever
- severe chest pain
- extreme headache

## Modify

```text
app/services/symptom_extractor.py
```

## Add Severity Keywords

```python
SEVERITY_KEYWORDS = {
    "mild": "LOW",
    "slight": "LOW",
    "moderate": "MEDIUM",
    "severe": "HIGH",
    "extreme": "HIGH",
    "unbearable": "HIGH"
}
```

## Add Bangla Severity Support

Examples:

```python
BANGLA_SEVERITY = {
    "হালকা": "LOW",
    "তীব্র": "HIGH",
    "অনেক ব্যথা": "HIGH"
}
```

## Output Structure

Return:

```json
{
  "symptoms": [],
  "severity": "HIGH"
}
```

## Constraints

- Preserve multilingual support
- Do NOT use heavy NLP models
- Use regex/lightweight processing only

---

# TASK 4 — Add Duration Detection

## Goal

Extract symptom duration information.

Examples:
- 2 days
- 3 weeks
- since yesterday
- chronic cough

## Modify

```text
app/services/symptom_extractor.py
```

## Add Regex Patterns

```python
DURATION_PATTERNS = [
    r"\\d+\\s*day",
    r"\\d+\\s*days",
    r"\\d+\\s*week",
    r"\\d+\\s*weeks",
    r"since yesterday",
    r"chronic"
]
```

## Output

```json
{
  "duration": "3 days"
}
```

## Requirements

Inject extracted duration into:
- prompt builder
- follow-up generation

---

# TASK 5 — Add Confidence Threshold Filtering

## Goal

Reduce weak noisy disease predictions.

## Modify

```text
app/services/predictor_service.py
```

## Requirements

1. Add configurable threshold:

```python
MIN_CONFIDENCE = 0.05
```

2. Filter predictions below threshold
3. Return only meaningful predictions
4. If all predictions are weak:
   - return empty prediction list
   - encourage more follow-up questioning

## Constraints

- Preserve predict_proba support
- Do NOT retrain model
- Preserve current JSON format

---

# TASK 6 — Expand Emergency Detection Rules

## Goal

Improve medical emergency detection.

## Modify

```text
app/services/emergency_engine.py
app/services/medical_rules.py
```

## Add Detection Rules

### Stroke Indicators

- slurred speech
- facial drooping
- one-sided weakness
- sudden confusion

### Allergic Emergencies

- throat swelling
- severe rash
- breathing difficulty

### Diabetic Emergencies

- excessive thirst
- unconsciousness
- severe confusion

### Neurological Emergencies

- seizures
- paralysis
- loss of consciousness

## Requirements

- Hardcoded rules must override LLM reasoning
- Preserve urgency levels:
  - EMERGENCY
  - HIGH
  - MEDIUM
  - LOW

## Constraints

- Do NOT rely on LLM for emergency detection

---

# TASK 7 — Add Age + Gender Context

## Goal

Improve medical reasoning using demographic context.

## Modify

```text
app/models/schemas.py
app/routes/chat.py
```

## Update ChatRequest

```python
class ChatRequest(BaseModel):
    message: str
    conversation_id: str

    age: int | None = None
    gender: str | None = None

    chronic_conditions: list[str] = []
```

## Requirements

Inject into prompts:
- age
- gender
- chronic conditions

## Constraints

- Preserve backward compatibility
- Existing clients without these fields must still work

---

# TASK 8 — Add Safer Advice Engine

## Goal

Provide supportive care guidance WITHOUT dangerous prescription behavior.

## Create

```text
app/services/advice_engine.py
```

## Requirements

Return:
- hydration advice
- rest guidance
- warning signs
- doctor consultation guidance

## IMPORTANT

Do NOT:
- prescribe antibiotics
- give dosage instructions
- recommend steroids
- recommend insulin changes
- provide chronic medication advice

## Example Output

```json
{
  "advice": [
    "Stay hydrated",
    "Monitor symptoms",
    "Seek medical care if symptoms worsen"
  ]
}
```

---

# TASK 9 — Improve Prompt Injection Context

## Goal

Improve LLM reasoning quality.

## Modify

```text
app/services/prompt_service.py
app/services/llm_service.py
```

## Inject Additional Context

Prompt templates should receive:

- duration
- severity
- age
- gender
- chronic conditions
- conversation memory
- previous predictions

## Example

```python
prompt_template.format(
    user_message=user_message,
    symptoms=symptoms,
    severity=severity,
    duration=duration,
    memory=memory,
    age=age,
    gender=gender
)
```

## Constraints

- Preserve JSON-only responses
- Preserve existing prompt structure

---

# TASK 10 — Add JSON Validation + Fallback Handling

## Goal

Prevent malformed LLM JSON responses from crashing APIs.

## Modify

```text
app/services/llm_service.py
```

## Requirements

1. Validate LLM JSON using:

```python
json.loads()
```

2. Add fallback recovery behavior
3. If parsing fails:
   - return safe fallback response
   - log parsing failure

## Constraints

- Preserve existing response schema
- Do NOT expose raw LLM errors to users

---

# TASK 11 — Add Logging

## Goal

Improve debugging and VPS monitoring.

## Requirements

Add logging for:
- prediction flow
- upload failures
- OCR failures
- Ollama failures
- JSON parsing failures
- emergency detections

## Constraints

- Use Python logging module
- Avoid excessive verbose logs

---

# TASK 12 — Add Ollama Timeout Protection

## Goal

Prevent hanging requests.

## Modify

```text
app/services/llm_service.py
```

## Requirements

1. Add request timeout handling
2. Add retry protection
3. Return graceful fallback responses if Ollama fails

## Constraints

- Preserve existing API structure
- Avoid crashing FastAPI workers

---

# IMPORTANT ENGINEERING RULES

## DO NOT

- rewrite the whole project
- refactor unrelated modules
- retrain the ML model
- rename APIs
- remove disclaimer logic
- break frontend compatibility

## PRESERVE

- FastAPI architecture
- modular service-layer pattern
- existing endpoints
- existing response schemas
- multilingual support
- JSON-only LLM responses

---

# PRIORITY ORDER

Implement tasks in this order:

1. Conversation memory
2. Severity extraction
3. Duration extraction
4. Confidence threshold
5. Emergency rules
6. Age/gender context
7. Advice engine
8. Logging + timeout handling

---

# FINAL OBJECTIVE

Improve:
- conversational continuity
- medical reasoning quality
- extraction quality
- backend robustness
- deployment safety

WITHOUT changing the overall IASIS AI architecture.

