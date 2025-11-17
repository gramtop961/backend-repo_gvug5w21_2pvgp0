import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone, date
import requests

from database import create_document, get_documents, db
from schemas import User, Mantra, JournalEntry, OracleQuery, MeditationSession, Course, Progress

app = FastAPI(title="WonderLens Chronicles API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"name": "WonderLens Chronicles API", "status": "ok"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, "name") else "Unknown"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# --------- Helper: OpenAI calls with graceful fallback ---------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def openai_chat(messages: List[dict], temperature: float = 0.7, max_tokens: int = 300) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


# --------- API: Users / Onboarding ---------
class OnboardingPayload(BaseModel):
    email: str
    name: str
    stage: str  # Awakening, Healing, Embodiment, Manifestation, Communion
    locale: Optional[str] = "en"
    avatar_url: Optional[str] = None


@app.post("/api/onboarding")
def onboarding(payload: OnboardingPayload):
    # Store as a user profile document
    user = User(
        email=payload.email,
        name=payload.name,
        stage=payload.stage,
        locale=payload.locale,
        avatar_url=payload.avatar_url,
    )
    try:
        # Upsert via simple approach: if exists, insert another with updated timestamp; for MVP, this is ok
        create_document("user", user)
        return {"ok": True, "message": "Profile saved", "stage": user.stage}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/{email}")
def get_user(email: str):
    try:
        docs = get_documents("user", {"email": email}, limit=1)
        if not docs:
            raise HTTPException(status_code=404, detail="User not found")
        doc = docs[0]
        # Convert ObjectId
        doc["_id"] = str(doc["_id"]) if "_id" in doc else None
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- API: Daily Mantra Generator ---------
class MantraRequest(BaseModel):
    email: str
    mood: Optional[str] = None
    stage: Optional[str] = None
    journal_theme: Optional[str] = None


@app.post("/api/mantra")
def generate_mantra(req: MantraRequest):
    today = date.today().isoformat()
    prompt = (
        "Create a daily mantra in African spiritual tone.\n"
        f"Inputs: mood={req.mood or 'balanced'}, stage={req.stage or 'Awakening'}, "
        f"recent_journal_theme={req.journal_theme or 'growth'}.\n"
        "Output: 1–2 line mantra + one-sentence meaning."
    )

    messages = [
        {
            "role": "system",
            "content": "You are a wise African spiritual guide blending Yoruba, Kikuyu, and Kemetian wisdom in a respectful, inclusive way.",
        },
        {"role": "user", "content": prompt},
    ]
    text = openai_chat(messages, temperature=0.8, max_tokens=200)

    if not text:
        # Fallback mantra
        text = (
            "I walk with Ngai in quiet power; Ashe flows through my breath.\n"
            "Meaning: Today I trust the current and let my steps be guided."
        )

    # Try to split into mantra and meaning
    parts = text.split("Meaning:")
    mantra_text = parts[0].strip()
    meaning = parts[1].strip() if len(parts) > 1 else "A reminder to align with breath, ancestors, and purpose."

    doc = Mantra(
        email=req.email,
        date=today,
        mood=req.mood,
        stage=req.stage,
        journal_theme=req.journal_theme,
        text=mantra_text,
        meaning=meaning,
    )
    try:
        create_document("mantra", doc)
        return {"ok": True, "date": today, "text": mantra_text, "meaning": meaning}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- API: Journal ---------
class JournalRequest(BaseModel):
    email: str
    content: str


@app.post("/api/journal")
def add_journal(req: JournalRequest):
    # Basic heuristic sentiment if no AI
    content_lower = req.content.lower()
    positive_words = ["grateful", "love", "peace", "flow", "joy", "light"]
    negative_words = ["fear", "anger", "sad", "doubt", "resistance", "pain"]
    score = 0
    score += sum(w in content_lower for w in positive_words)
    score -= sum(w in content_lower for w in negative_words)
    sentiment = max(-1.0, min(1.0, score / 5.0))

    # Optionally call OpenAI for theme
    theme = None
    summary = openai_chat(
        [
            {"role": "system", "content": "Summarize the central theme in 6 words or fewer."},
            {"role": "user", "content": req.content},
        ],
        temperature=0.2,
        max_tokens=40,
    )
    if summary:
        theme = summary

    entry = JournalEntry(email=req.email, content=req.content, mood=None, sentiment_score=sentiment, tags=[theme] if theme else [])
    try:
        create_document("journalentry", entry)
        return {"ok": True, "sentiment_score": sentiment, "theme": theme}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- API: Oracle ---------
class OracleRequest(BaseModel):
    email: str
    input_text: str


@app.post("/api/oracle")
def oracle(req: OracleRequest):
    prompt = (
        "Interpret this using African spiritual wisdom (respectfully blending Yoruba, Kikuyu, and Kemetian principles).\n"
        "Explain the metaphysical cause and the lesson shown. Offer a gentle action step.\n\n"
        f"Input: {req.input_text}"
    )
    messages = [
        {"role": "system", "content": "You are The Lens — a compassionate spiritual guide."},
        {"role": "user", "content": prompt},
    ]
    interpretation = openai_chat(messages, temperature=0.7, max_tokens=350)
    if not interpretation:
        interpretation = (
            "Your dream reflects a shift from resistance to alignment. The cause is inner doubt meeting ancestral guidance."
            " Lesson: trust your breath and take one small act of devotion today."
        )

    try:
        create_document("oraclequery", OracleQuery(email=req.email, input_text=req.input_text, interpretation=interpretation))
        return {"ok": True, "interpretation": interpretation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- API: Meditation Sessions ---------
class MeditationRequest(BaseModel):
    email: str
    environment: str
    minutes: int


@app.post("/api/meditation")
def meditation(req: MeditationRequest):
    try:
        create_document("meditationsession", MeditationSession(email=req.email, environment=req.environment, minutes=req.minutes))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------- API: Courses & Progress ---------
COURSES: List[Course] = [
    Course(slug="law-of-vibration", title="Understanding the Law of Vibration", summary="How energy patterns shape reality.", level="Initiate"),
    Course(slug="african-energy-centers", title="African Energy Centers Explained", summary="Centers of power and breath.", level="Seer"),
    Course(slug="inner-child-ancestral", title="Healing the Inner Child through Ancestral Wisdom", summary="Reparenting with lineage support.", level="Sage"),
]


@app.get("/api/courses")
def list_courses():
    return [c.model_dump() for c in COURSES]


class ProgressRequest(BaseModel):
    email: str
    course_slug: str
    lesson_id: str


@app.post("/api/progress")
def track_progress(req: ProgressRequest):
    # Simple insert log; viewer can aggregate
    try:
        create_document(
            "progress",
            Progress(email=req.email, course_slug=req.course_slug, completed_lessons=[req.lesson_id], badge=None),
        )
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backwards compatible hello
@app.get("/api/hello")
def hello():
    return {"message": "WonderLens backend is alive"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
