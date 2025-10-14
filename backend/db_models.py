"""
SQLAlchemy database models
"""
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    """User model - synced with Auth0"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    auth0_id = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    name = Column(String)
    roles = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))

    # Relationships
    cases = relationship("Case", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")

    __table_args__ = (
        Index("ix_users_auth0_id", "auth0_id"),
        Index("ix_users_email", "email"),
    )


class Case(Base):
    """Generated case model"""
    __tablename__ = "cases"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.auth0_id"), nullable=False)

    # Case metadata
    student_name = Column(String, nullable=False)
    grade = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)
    disorders = Column(JSON, nullable=False)  # List of disorders
    model_used = Column(String, nullable=False)
    characteristics = Column(Text)

    # Case content
    background_info = Column(JSON)  # medical_history, parent_concerns, teacher_concerns
    annual_goals = Column(JSON)  # List of goals
    session_notes = Column(JSON)  # List of session notes
    full_content = Column(Text)  # Full markdown content

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    batch_id = Column(String, nullable=True)  # For multiple case generations

    # Relationships
    user = relationship("User", back_populates="cases")
    feedback = relationship("Feedback", back_populates="case")

    __table_args__ = (
        Index("ix_cases_user_id", "user_id"),
        Index("ix_cases_created_at", "created_at"),
        Index("ix_cases_grade", "grade"),
        Index("ix_cases_batch_id", "batch_id"),
    )


class Feedback(Base):
    """Feedback on generated cases"""
    __tablename__ = "feedback"

    id = Column(String, primary_key=True, default=generate_uuid)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.auth0_id"), nullable=False)

    # Ratings (1-5 scale)
    ratings = Column(JSON, nullable=False)  # Dict of rating categories and scores

    # Feedback content
    category = Column(String, nullable=False)
    detailed_feedback = Column(Text)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    case = relationship("Case", back_populates="feedback")
    user = relationship("User", back_populates="feedback")

    __table_args__ = (
        Index("ix_feedback_case_id", "case_id"),
        Index("ix_feedback_user_id", "user_id"),
        Index("ix_feedback_category", "category"),
    )


class AuditLog(Base):
    """Audit log for compliance and security"""
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.auth0_id"), nullable=False)

    # Action details
    action = Column(String, nullable=False)
    resource = Column(String, nullable=False)
    resource_id = Column(String, nullable=True)

    # Request metadata
    ip_address = Column(String)
    user_agent = Column(String)
    extra_metadata = Column(JSON)  # Additional context

    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_logs_user_id", "user_id"),
        Index("ix_audit_logs_action", "action"),
        Index("ix_audit_logs_timestamp", "timestamp"),
        Index("ix_audit_logs_resource", "resource"),
    )
