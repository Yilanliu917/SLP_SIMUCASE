"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DisorderType(str, Enum):
    SPEECH_SOUND = "Speech Sound Disorder"
    ARTICULATION = "Articulation Disorders"
    PHONOLOGICAL = "Phonological Disorders"
    LANGUAGE = "Language Disorders"
    RECEPTIVE = "Receptive Language Disorders"
    EXPRESSIVE = "Expressive Language Disorders"
    PRAGMATICS = "Pragmatics"
    FLUENCY = "Fluency"
    CAS = "Childhood Apraxia of Speech"


class ModelType(str, Enum):
    LLAMA32 = "Llama3.2"
    QWEN_7B = "Qwen 2.5 7B"
    QWEN_32B = "Qwen 2.5 32B"
    DEEPSEEK = "DeepSeek R1 32B"
    GPT4O = "GPT-4o"
    GEMINI = "Gemini 2.5 Pro"
    CLAUDE_OPUS = "Claude 3 Opus"
    CLAUDE_SONNET = "Claude 3.5 Sonnet"


class CaseGenerationRequest(BaseModel):
    """Request model for case generation"""
    grade: str = Field(..., description="Grade level (e.g., '1st Grade', 'Kindergarten')")
    disorders: List[DisorderType] = Field(..., description="List of disorders")
    model: ModelType = Field(default=ModelType.LLAMA32, description="AI model to use")
    population_spec: Optional[str] = Field(None, description="Special population characteristics")
    characteristics: Optional[str] = Field(None, description="Specific characteristics for the case")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "grade": "1st Grade",
                "disorders": ["Articulation Disorders"],
                "model": "Llama3.2",
                "population_spec": "second language learner",
                "characteristics": "difficulty with /s/ sound"
            }
        }
    )


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


class CaseGenerationResponse(BaseModel):
    """Response model for case generation"""
    case_id: str
    student_profile: StudentProfile
    annual_goals: List[str]
    latest_session_notes: List[str]
    generated_at: datetime
    model_used: str
    user_id: str

    model_config = ConfigDict(from_attributes=True)


class CaseListItem(BaseModel):
    """Model for case list items"""
    case_id: str
    student_name: str
    grade: str
    disorders: List[str]
    created_at: datetime
    user_id: str


class FeedbackSubmission(BaseModel):
    """Model for feedback submission"""
    case_id: str
    ratings: Dict[str, int] = Field(..., description="Rating scores (1-5)")
    category: str
    detailed_feedback: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "case_id": "case_123",
                "ratings": {
                    "clinical_accuracy": 4,
                    "age_appropriateness": 5,
                    "goal_quality": 4
                },
                "category": "Clinical Accuracy",
                "detailed_feedback": "Goals are well-structured but could include more specific metrics"
            }
        }
    )


class UserActivityLog(BaseModel):
    """Model for user activity logging"""
    user_id: str
    action: str
    resource: str
    resource_id: Optional[str] = None
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class AuditLog(BaseModel):
    """Model for audit log entries"""
    log_id: str
    user_id: str
    action: str
    resource: str
    resource_id: Optional[str] = None
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class UsageAnalytics(BaseModel):
    """Model for usage analytics"""
    total_cases: int
    total_users: int
    cases_by_model: Dict[str, int]
    cases_by_disorder: Dict[str, int]
    cases_by_grade: Dict[str, int]
    daily_activity: List[Dict[str, Any]]


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
