"""
Database Schemas for WonderLens Chronicles

Each Pydantic model represents a MongoDB collection.
Collection name is the lowercase of the class name.
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

# Core user profile
class User(BaseModel):
    email: EmailStr = Field(..., description="Unique email for the user")
    name: str = Field(..., description="Display name")
    stage: str = Field(
        "Awakening",
        description="Current spiritual stage: Awakening, Healing, Embodiment, Manifestation, Communion",
    )
    avatar_url: Optional[str] = Field(None, description="Profile avatar URL")
    locale: Optional[str] = Field("en", description="Language preference")

# Daily mantra record
class Mantra(BaseModel):
    email: EmailStr = Field(..., description="Owner email")
    date: str = Field(..., description="ISO date (YYYY-MM-DD)")
    mood: Optional[str] = Field(None, description="User-reported mood for the day")
    stage: Optional[str] = Field(None, description="Stage used for generation")
    journal_theme: Optional[str] = Field(None, description="Recent journal theme summary")
    text: str = Field(..., description="Generated mantra text")
    meaning: Optional[str] = Field(None, description="Short explanation/meaning of mantra")

# Journal entry
class JournalEntry(BaseModel):
    email: EmailStr = Field(...)
    content: str = Field(..., description="Full journal text or transcript")
    mood: Optional[str] = Field(None, description="Detected or self-reported mood")
    sentiment_score: Optional[float] = Field(
        None, description="-1 to 1 sentiment score"
    )
    tags: Optional[List[str]] = Field(default_factory=list)

# Oracle queries / interpretations
class OracleQuery(BaseModel):
    email: EmailStr = Field(...)
    input_text: str = Field(..., description="Dreams, symbols, or life events")
    interpretation: Optional[str] = None

# Meditation session logs (simple)
class MeditationSession(BaseModel):
    email: EmailStr
    environment: str = Field(..., description="forest, mt-kenya, desert-temple")
    minutes: int = Field(..., ge=1, le=240)

# Course content (static placeholder structure)
class Course(BaseModel):
    slug: str
    title: str
    summary: str
    level: str = Field("Initiate")

# Progress tracking
class Progress(BaseModel):
    email: EmailStr
    course_slug: str
    completed_lessons: List[str] = Field(default_factory=list)
    badge: Optional[str] = None

