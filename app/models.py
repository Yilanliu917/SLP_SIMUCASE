"""
Pydantic models and global state management
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# --- PYDANTIC MODELS ---
class BackgroundInfo(BaseModel):
    medical_history: str
    parent_concerns: str
    teacher_concerns: str

class StudentProfile(BaseModel):
    name: str
    age: int
    grade_level: str
    gender: str
    background: BackgroundInfo

class SimuCaseFile(BaseModel):
    student_profile: StudentProfile
    annual_goals: List[str]
    latest_session_notes: List[str]

# --- GLOBAL STATE ---
current_case_data = {
    "content": None,
    "case_id": None,
    "metadata": None
}

multiple_cases_batch = {
    "cases": [],
    "batch_id": None,
    "timestamp": None
}

group_session_data = {
    "session_id": None,
    "members": [],
    "timestamp": None
}

generation_control = {
    "should_stop": False
}
