"""
0211hr — Full Backend API
Powers HR Interview, Speaking Exam, and Candidate Database on the website.

Deploy to Railway:
  1. Push backend/ folder to GitHub
  2. Connect to railway.app → set GOOGLE_API_KEY env var
  3. Copy your Railway URL into the frontend APP_URL constant
"""

from __future__ import annotations

import json, os, re, uuid, time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="0211hr API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(500, "GOOGLE_API_KEY not set on server.")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.4,
        google_api_key=api_key,
    )

# ── CANDIDATE DATABASE ────────────────────────────────────────────────────────

DB_FILE = Path("candidate_database.json")

def load_db() -> List[Dict]:
    if not DB_FILE.exists():
        return []
    try:
        return json.loads(DB_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_db(data: List[Dict]):
    DB_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def upsert_candidate(session_id: str, name: str, tool: str, scores: Dict, evaluations: List, meta: Dict):
    db = load_db()
    entry = next((e for e in db if e.get("session_id") == session_id), None)
    if not entry:
        entry = {
            "session_id": session_id,
            "name": name or "Anonymous",
            "tool": tool,
            "started_at": datetime.utcnow().isoformat(),
            "sessions": []
        }
        db.append(entry)
    entry["name"] = name or entry.get("name", "Anonymous")
    entry["last_updated"] = datetime.utcnow().isoformat()
    entry["final_scores"] = scores
    entry["evaluations"] = evaluations
    entry["meta"] = meta
    save_db(db)
    return entry

# ── SYSTEM PROMPTS ────────────────────────────────────────────────────────────

HR_SYSTEM = """You are Alex, a professional AI HR interviewer for 0211hr.
Conduct a structured interview across 4 phases: intro, competency, scenario, closing.
Rules:
- Keep replies to 3-5 sentences max. One question at a time.
- After user responds (not on first message), evaluate their answer briefly then ask next question.
- Track which phase you are in based on conversation length.
- Phase flow: intro (1-2 exchanges) → competency (3-4 exchanges) → scenario (2-3 exchanges) → closing (1 exchange)

Always respond with ONLY valid JSON, no markdown, no extra text:
{
  "message": "your spoken reply",
  "scores": {
    "star": <0-10>,
    "technical": <0-10>,
    "soft_skills": <0-10>,
    "cultural_fit": <0-10>,
    "consistency": <0-10>,
    "confidence": <0-10>
  },
  "phase": "intro|competency|scenario|closing",
  "feedback": "brief internal note about candidate's last answer",
  "interview_complete": false
}
Set interview_complete to true only during closing phase after final question answered.
First message: warm greeting, ask for name and target role. scores can all be 0 on first message."""

SPEAKING_SYSTEM = """You are Alex, a professional AI speaking examiner for 0211hr.
Conduct a 3-part IELTS-style oral exam.
- Part 1 (Narrative): Ask candidate to describe a past experience in detail.
- Part 2 (Debate): Present a statement and ask for their opinion with arguments.
- Part 3 (Reading+Summary): Give a short 80-word passage inline, ask candidate to summarize key points.

Rules:
- One part per 1-2 exchanges. Move forward naturally.
- After each response (not first), score it then transition to next part or conclude.

Always respond with ONLY valid JSON, no markdown:
{
  "message": "your spoken reply to candidate",
  "scores": {
    "fluency": <0-10>,
    "grammar": <0-10>,
    "vocabulary": <0-10>,
    "cohesion": <0-10>,
    "pronunciation": <0-10>
  },
  "ielts_band": <0.0-9.0>,
  "part": <1|2|3>,
  "feedback": "brief evaluator note",
  "exam_complete": false
}
Set exam_complete to true after Part 3 is scored. First message: greet, explain structure, begin Part 1. scores can all be 0 on first message."""

SPEAKING_SYSTEM_LANG = {
    "es": SPEAKING_SYSTEM.replace("Conduct a 3-part IELTS-style oral exam.", "Conduct a 3-part IELTS-style oral exam IN SPANISH. All your messages must be in Spanish."),
    "fr": SPEAKING_SYSTEM.replace("Conduct a 3-part IELTS-style oral exam.", "Conduct a 3-part IELTS-style oral exam IN FRENCH. All your messages must be in French."),
    "ar": SPEAKING_SYSTEM.replace("Conduct a 3-part IELTS-style oral exam.", "Conduct a 3-part IELTS-style oral exam IN ARABIC. All your messages must be in Arabic."),
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def parse_llm_json(text: str) -> Dict:
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(clean)
    except Exception:
        # Try to extract JSON object
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"message": text, "scores": {}, "phase": "intro", "part": 1,
            "ielts_band": 5.0, "feedback": "", "interview_complete": False, "exam_complete": False}

def build_messages(system: str, history: List[Dict], new_msg: str):
    msgs = [SystemMessage(content=system)]
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=new_msg))
    return msgs

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────

class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class HRRequest(BaseModel):
    session_id: str
    history: List[HistoryItem] = []
    message: str
    candidate_name: Optional[str] = ""

class SpeakingRequest(BaseModel):
    session_id: str
    history: List[HistoryItem] = []
    message: str
    language: Literal["en", "es", "fr", "ar"] = "en"

class HRResponse(BaseModel):
    message: str
    scores: Dict[str, Any]
    phase: str
    feedback: str
    interview_complete: bool
    session_id: str

class SpeakingResponse(BaseModel):
    message: str
    scores: Dict[str, Any]
    ielts_band: float
    part: int
    feedback: str
    exam_complete: bool
    session_id: str

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/interview/hr", response_model=HRResponse)
def hr_interview(req: HRRequest):
    llm = get_llm()
    system = HR_SYSTEM
    history = [{"role": m.role, "content": m.content} for m in req.history]
    messages = build_messages(system, history, req.message)

    try:
        result = llm.invoke(messages)
        parsed = parse_llm_json(result.content)
    except Exception as e:
        raise HTTPException(500, str(e))

    scores = parsed.get("scores", {})
    complete = bool(parsed.get("interview_complete", False))

    # Save to candidate DB
    upsert_candidate(
        session_id=req.session_id,
        name=req.candidate_name or "",
        tool="hr",
        scores=scores,
        evaluations=history + [{"role": "assistant", "content": parsed.get("message", "")}],
        meta={"phase": parsed.get("phase", "intro"), "complete": complete}
    )

    return HRResponse(
        message=parsed.get("message", ""),
        scores=scores,
        phase=parsed.get("phase", "intro"),
        feedback=parsed.get("feedback", ""),
        interview_complete=complete,
        session_id=req.session_id,
    )


@app.post("/interview/speaking", response_model=SpeakingResponse)
def speaking_exam(req: SpeakingRequest):
    llm = get_llm()
    system = SPEAKING_SYSTEM_LANG.get(req.language, SPEAKING_SYSTEM)
    history = [{"role": m.role, "content": m.content} for m in req.history]
    messages = build_messages(system, history, req.message)

    try:
        result = llm.invoke(messages)
        parsed = parse_llm_json(result.content)
    except Exception as e:
        raise HTTPException(500, str(e))

    scores = parsed.get("scores", {})
    complete = bool(parsed.get("exam_complete", False))
    band = float(parsed.get("ielts_band", 5.0))

    upsert_candidate(
        session_id=req.session_id,
        name="",
        tool="speaking",
        scores=scores,
        evaluations=history + [{"role": "assistant", "content": parsed.get("message", "")}],
        meta={"part": parsed.get("part", 1), "ielts_band": band,
              "language": req.language, "complete": complete}
    )

    return SpeakingResponse(
        message=parsed.get("message", ""),
        scores=scores,
        ielts_band=band,
        part=int(parsed.get("part", 1)),
        feedback=parsed.get("feedback", ""),
        exam_complete=complete,
        session_id=req.session_id,
    )


@app.get("/candidates")
def get_candidates():
    db = load_db()
    return {"candidates": db, "total": len(db)}


@app.delete("/candidates/{session_id}")
def delete_candidate(session_id: str):
    db = load_db()
    db = [e for e in db if e.get("session_id") != session_id]
    save_db(db)
    return {"ok": True}

