"""
0211hr MeetBot Backend v1.2
Recall.ai bot yönetimi + transkript + Gemini evaluation
"""

from __future__ import annotations
import hashlib, hmac, json, logging, os, re, uuid, httpx
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meetbot")

# ── CONFIG ────────────────────────────────────────────────────────
RECALL_API_KEY  = os.getenv("RECALL_API_KEY", "")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
RECALL_BASE     = "https://us-east-1.recall.ai/api/v1"
WEBHOOK_SECRET  = os.getenv("RECALL_WEBHOOK_SECRET", "")

# ── HTTP CLIENT (lifespan ile yönetiliyor) ────────────────────────
http_client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=30)
    logger.info("HTTP client başlatıldı.")
    yield
    await http_client.aclose()
    logger.info("HTTP client kapatıldı.")

# ── APP ──────────────────────────────────────────────────────────
app = FastAPI(
    title="0211hr MeetBot API",
    version="1.2.0",
    lifespan=lifespan,
)

_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in _origins.split(",")] if _origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SESSION STORE ─────────────────────────────────────────────────
SESSIONS: Dict[str, Dict] = {}
BOT_TO_SESSION: Dict[str, str] = {}

# ── LLM ──────────────────────────────────────────────────────────
def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not GOOGLE_API_KEY:
        raise HTTPException(500, "GOOGLE_API_KEY not set.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GOOGLE_API_KEY,
    )

# ── PROMPTS ───────────────────────────────────────────────────────
EVAL_SYSTEM = """You are a senior HR analyst. Evaluate this job interview transcript.
Return ONLY valid JSON — no markdown, no extra text:
{
  "scores": {
    "star": 0-10,
    "technical": 0-10,
    "soft_skills": 0-10,
    "cultural_fit": 0-10,
    "consistency": 0-10,
    "confidence": 0-10
  },
  "overall_assessment": "hire|maybe|pass",
  "brief_summary": "3-4 sentence summary of the candidate",
  "green_flags": ["positive signal 1", "positive signal 2"],
  "red_flags": ["concern 1", "concern 2"],
  "keywords": ["keyword1", "keyword2"],
  "interviewer_coach": "one targeted follow-up question for the weakest dimension",
  "recommended_positions": [
    {"title": "Job Title", "reason": "why this fits the candidate"}
  ]
}"""

# ── HELPERS ───────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_json(text: str) -> Dict:
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                logger.warning("JSON parse fallback da başarısız.")
    return {}

def parse_turns(transcript: List[Dict]) -> List[Dict]:
    """Recall transkript listesini turn listesine çevir."""
    turns = []
    for item in transcript:
        speaker = item.get("speaker") or "Unknown"
        words   = item.get("words") or []
        # words hem list[dict] hem string olabilir — savunmacı parse
        if isinstance(words, list):
            text = " ".join(
                w.get("text", "") if isinstance(w, dict) else str(w)
                for w in words
            ).strip()
        else:
            text = str(words).strip()
        if text:
            turns.append({
                "speaker": speaker,
                "text": text,
                "timestamp": item.get("start_timestamp") or _now_iso(),
            })
    return turns

# ── WEBHOOK SIGNATURE ─────────────────────────────────────────────
def verify_webhook_signature(body: bytes, signature: str) -> bool:
    """
    Recall.ai webhook HMAC-SHA256 doğrulaması.
    Header formatı: x-recall-signature: sha256=<hex_digest>
    """
    if not WEBHOOK_SECRET:
        logger.warning("RECALL_WEBHOOK_SECRET ayarlanmamış — doğrulama atlanıyor.")
        return True
    # "sha256=" prefix'ini temizle
    sig_clean = signature.removeprefix("sha256=").strip()
    expected  = hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig_clean)

# ── RECALL API ────────────────────────────────────────────────────
def recall_headers() -> Dict:
    if not RECALL_API_KEY:
        raise HTTPException(500, "RECALL_API_KEY not set.")
    return {
        "Authorization": f"Token {RECALL_API_KEY}",
        "Content-Type": "application/json",
    }

async def recall_post(path: str, payload: Dict) -> Dict:
    r = await http_client.post(
        f"{RECALL_BASE}{path}",
        headers=recall_headers(),
        json=payload,
    )
    if r.status_code not in (200, 201):
        logger.error("Recall POST %s → %s: %s", path, r.status_code, r.text)
        raise HTTPException(400, f"Recall error: {r.text}")
    return r.json()

async def recall_get_json(path: str) -> Any:
    r = await http_client.get(
        f"{RECALL_BASE}{path}",
        headers=recall_headers(),
    )
    if r.status_code != 200:
        logger.error("Recall GET %s → %s: %s", path, r.status_code, r.text)
        raise HTTPException(400, f"Recall error: {r.text}")
    return r.json()

async def recall_create_bot(meeting_url: str) -> Dict:
    return await recall_post("/bot/", {
        "meeting_url": meeting_url,
        "bot_name": "0211hr Assistant",
        "transcription_options": {"provider": "assembly_ai"},
        "recording_mode": "speaker_view",
        "chat": {
            "on_bot_join": {
                "send_to": "everyone",
                "message": "Merhaba! Ben 0211hr AI asistanıyım. Bu görüşmeyi değerlendireceğim.",
            }
        },
    })

async def recall_get_transcript(bot_id: str) -> List[Dict]:
    try:
        data = await recall_get_json(f"/bot/{bot_id}/transcript/")
        raw  = data if isinstance(data, list) else data.get("results", [])
        return parse_turns(raw)
    except Exception as e:
        logger.warning("Transkript alınamadı (bot_id=%s): %s", bot_id, e)
        return []

async def recall_stop_bot(bot_id: str):
    try:
        await recall_post(f"/bot/{bot_id}/leave_call/", {})
    except Exception as e:
        logger.warning("Bot durdurulamadı (bot_id=%s): %s", bot_id, e)

# ── EVALUATION ────────────────────────────────────────────────────
async def run_evaluation(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        logger.warning("Evaluation: session bulunamadı — %s", session_id)
        return

    turns = session.get("turns", [])
    if not turns:
        logger.info("Evaluation: transkript boş — %s", session_id)
        session["status"] = "complete"
        return

    transcript_text = "\n".join(
        f"{t['speaker'].upper()}: {t['text']}" for t in turns
    )

    prompt = (
        f"Candidate: {session.get('candidate_name', 'Unknown')}\n"
        f"Target Role: {session.get('target_role', 'Unknown')}\n\n"
        f"Interview Transcript:\n{transcript_text}\n\n"
        "Evaluate this interview and return the JSON report."
    )

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        llm    = get_llm()
        result = llm.invoke([
            SystemMessage(content=EVAL_SYSTEM),
            HumanMessage(content=prompt),
        ])
        report = parse_json(result.content)
        if not report:
            raise ValueError("LLM boş JSON döndürdü.")
        session["evaluation"] = report
        session["scores"]     = report.get("scores", {})
        session["status"]     = "complete"
        logger.info("Evaluation tamamlandı — %s | assessment: %s",
                    session_id, report.get("overall_assessment"))
    except Exception as e:
        logger.error("Evaluation hatası — %s: %s", session_id, e)
        session["evaluation_error"] = str(e)
        session["status"]           = "complete"

# ── REQUEST MODELS ────────────────────────────────────────────────
class BotStartReq(BaseModel):
    meeting_url:    str
    candidate_name: Optional[str] = "Anonymous"
    target_role:    Optional[str] = "Unknown"

# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.2.0",
        "sessions": len(SESSIONS),
        "recall_configured": bool(RECALL_API_KEY),
        "gemini_configured": bool(GOOGLE_API_KEY),
    }

# ── BOT START ─────────────────────────────────────────────────────
@app.post("/bot/start")
async def bot_start(req: BotStartReq):
    bot_data = await recall_create_bot(req.meeting_url)
    bot_id   = bot_data.get("id")
    if not bot_id:
        raise HTTPException(500, "Recall'dan bot ID alınamadı.")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "session_id":    session_id,
        "bot_id":        bot_id,
        "meeting_url":   req.meeting_url,
        "candidate_name":req.candidate_name,
        "target_role":   req.target_role,
        "status":        "active",
        "started_at":    _now_iso(),
        "ended_at":      None,
        "turns":         [],
        "scores":        {},
        "evaluation":    None,
        "evaluation_error": None,
    }
    BOT_TO_SESSION[bot_id] = session_id
    logger.info("Bot başlatıldı — session=%s bot=%s", session_id, bot_id)

    return {
        "session_id": session_id,
        "bot_id":     bot_id,
        "status":     "active",
        "message":    "Bot toplantıya katılıyor.",
    }

# ── BOT STATUS ────────────────────────────────────────────────────
@app.get("/bot/{session_id}/status")
async def bot_status(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session bulunamadı.")

    # Aktif session'da transkripti güncelle
    if session["status"] == "active":
        bot_id = session.get("bot_id")
        if bot_id:
            turns = await recall_get_transcript(bot_id)
            if turns:
                session["turns"] = turns

    return {
        "session_id":    session_id,
        "status":        session["status"],
        "candidate_name":session["candidate_name"],
        "target_role":   session["target_role"],
        "meeting_url":   session["meeting_url"],
        "started_at":    session["started_at"],
        "turn_count":    len(session["turns"]),
        "turns":         session["turns"][-20:],
        "scores":        session["scores"],
    }

# ── BOT STOP ──────────────────────────────────────────────────────
@app.post("/bot/{session_id}/stop")
async def bot_stop(session_id: str, background_tasks: BackgroundTasks):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session bulunamadı.")

    bot_id = session.get("bot_id")
    if bot_id:
        turns = await recall_get_transcript(bot_id)
        if turns:
            session["turns"] = turns
        await recall_stop_bot(bot_id)

    session["status"]   = "evaluating"
    session["ended_at"] = _now_iso()
    background_tasks.add_task(run_evaluation, session_id)
    logger.info("Bot durduruldu, evaluation başlatıldı — %s", session_id)

    return {
        "session_id": session_id,
        "status":     "evaluating",
        "message":    "Bot durduruldu, evaluation başlatıldı.",
    }

# ── WEBHOOK ───────────────────────────────────────────────────────
@app.post("/bot/webhook")
async def bot_webhook(request: Request, background_tasks: BackgroundTasks):
    raw_body  = await request.body()
    signature = request.headers.get("x-recall-signature", "")

    if not verify_webhook_signature(raw_body, signature):
        logger.warning("Webhook signature doğrulaması başarısız!")
        raise HTTPException(401, "Invalid webhook signature.")

    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Webhook: geçersiz JSON body")
        return {"ok": True}

    event      = body.get("event", "")
    data       = body.get("data", {})
    bot_id     = data.get("bot_id") or data.get("id", "")
    session_id = BOT_TO_SESSION.get(bot_id)

    if not session_id or session_id not in SESSIONS:
        return {"ok": True}

    session = SESSIONS[session_id]
    logger.info("Webhook: event=%s session=%s", event, session_id)

    if event == "bot.in_call_recording":
        session["status"] = "active"

    elif event == "transcript.data":
        speaker = data.get("speaker", "Unknown")
        words   = data.get("words", [])
        text    = " ".join(
            w.get("text", "") if isinstance(w, dict) else str(w)
            for w in words
        ).strip()
        if text:
            session["turns"].append({
                "speaker":   speaker,
                "text":      text,
                "timestamp": _now_iso(),
            })

    elif event in ("bot.done", "bot.fatal_error", "bot.left_call"):
        session["status"]   = "evaluating"
        session["ended_at"] = _now_iso()
        background_tasks.add_task(run_evaluation, session_id)

    return {"ok": True}

# ── REPORT ────────────────────────────────────────────────────────
@app.get("/bot/{session_id}/report")
async def bot_report(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session bulunamadı.")
    if session["status"] == "active":
        raise HTTPException(400, "Meeting henüz devam ediyor.")
    if session["status"] == "evaluating":
        return {"status": "evaluating", "message": "Evaluation devam ediyor."}

    return {
        "session_id":       session_id,
        "candidate_name":   session["candidate_name"],
        "target_role":      session["target_role"],
        "started_at":       session["started_at"],
        "ended_at":         session.get("ended_at"),
        "turn_count":       len(session["turns"]),
        "scores":           session["scores"],
        "evaluation":       session["evaluation"],
        "evaluation_error": session.get("evaluation_error"),
    }

# ── ALL SESSIONS ──────────────────────────────────────────────────
@app.get("/sessions")
def get_sessions():
    return {
        "sessions": [
            {
                "session_id":    s["session_id"],
                "candidate_name":s["candidate_name"],
                "target_role":   s["target_role"],
                "status":        s["status"],
                "started_at":    s["started_at"],
                "ended_at":      s.get("ended_at"),
                "turn_count":    len(s["turns"]),
                "scores":        s["scores"],
                "assessment":    (s.get("evaluation") or {}).get("overall_assessment"),
            }
            for s in SESSIONS.values()
        ],
        "total": len(SESSIONS),
    }

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id in SESSIONS:
        bot_id = SESSIONS[session_id].get("bot_id")
        if bot_id:
            BOT_TO_SESSION.pop(bot_id, None)
        del SESSIONS[session_id]
    return {"ok": True}
