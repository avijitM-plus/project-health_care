from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes import health, chat, analyze, report, session
import time
import logging
from collections import defaultdict

# --- Centralized logging configuration (Task 11) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- API metadata for auto-generated docs (deliverable #13) ---
app = FastAPI(
    title="IASIS AI Medical Assistant Server",
    version="1.0.0",
    description=(
        "AI-powered medical assistant backend that chats with patients, "
        "analyzes symptoms, predicts possible diseases using ML, parses "
        "medical reports (PDF/Image), and flags emergencies. "
        "All responses include medical disclaimers."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "Server health status"},
        {"name": "Chat", "description": "Conversational symptom triage"},
        {"name": "Reports", "description": "Medical report upload and analysis"},
        {"name": "Session", "description": "Session management and monitoring"},
    ],
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simple in-memory rate limiter (security requirement) ---
RATE_LIMIT = 30          # max requests
RATE_WINDOW = 60         # per N seconds
_request_log: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    # Prune old entries
    _request_log[client_ip] = [
        ts for ts in _request_log[client_ip] if now - ts < RATE_WINDOW
    ]
    if len(_request_log[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."},
        )
    _request_log[client_ip].append(now)
    return await call_next(request)


# --- Routers ---
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(analyze.router, tags=["Reports"])
app.include_router(report.router, tags=["Reports"])
app.include_router(session.router, tags=["Session"])


# --- Startup event: log memory configuration ---
@app.on_event("startup")
async def startup_event():
    from app.services.memory_service import memory_service
    stats = memory_service.stats()
    logger.info(
        f"IASIS AI v2.0 starting — "
        f"Memory config: max_sessions={stats['max_sessions']}, "
        f"ttl={stats['ttl_seconds']}s"
    )


@app.get("/")
async def root():
    return {"message": "Welcome to IASIS AI API v2.0. Check /docs for API documentation."}
