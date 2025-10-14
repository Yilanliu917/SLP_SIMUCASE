"""
FastAPI Backend with RBAC for SLP SimuCase Generator
Production-ready with Auth0 integration, role-based access control, and audit logging
"""
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import List, Optional
import os

from auth import (
    verify_token,
    get_current_user,
    require_role,
    require_permission,
    User,
    Role
)
from database import init_db, get_db
from models import (
    CaseGenerationRequest,
    CaseGenerationResponse,
    UserActivityLog,
    AuditLog
)
from services import (
    case_generation_service,
    audit_service,
    analytics_service
)
from cache import get_cache, cache_response

# Configure logging
log_dir = os.getenv("LOG_DIR", "./logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info("Starting SLP SimuCase Backend API")
    await init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down SLP SimuCase Backend API")


# Initialize FastAPI app
app = FastAPI(
    title="SLP SimuCase Generator API",
    description="Production-ready backend with RBAC for Speech-Language Pathology case generation",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted Host Middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
)


# Health Check Endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for container orchestration"""
    return {
        "status": "healthy",
        "service": "slp-simucase-api",
        "version": "1.0.0"
    }


# Readiness Check
@app.get("/ready", tags=["Health"])
async def readiness_check(db=Depends(get_db)):
    """Readiness check - verifies database connectivity"""
    try:
        # Test database connection
        await db.execute("SELECT 1")
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


# ==================== RBAC Protected Endpoints ====================

# Role definitions:
# - admin: Full system access, user management
# - researcher: Can generate cases, access analytics, export data
# - clinician: Can generate cases, view own data
# - viewer: Read-only access


@app.get("/api/user/profile", tags=["User"])
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    await audit_service.log_activity(
        user_id=current_user.sub,
        action="view_profile",
        resource="user_profile"
    )
    return {
        "user_id": current_user.sub,
        "email": current_user.email,
        "name": current_user.name,
        "roles": current_user.roles,
        "permissions": current_user.permissions
    }


@app.post("/api/cases/generate",
          response_model=CaseGenerationResponse,
          tags=["Case Generation"])
async def generate_case(
    request: CaseGenerationRequest,
    current_user: User = Depends(require_role([Role.CLINICIAN, Role.RESEARCHER, Role.ADMIN]))
):
    """
    Generate a new case file
    Requires: clinician, researcher, or admin role
    """
    logger.info(f"User {current_user.email} requesting case generation")

    try:
        # Check rate limits
        cache = await get_cache()
        rate_limit_key = f"rate_limit:{current_user.sub}"
        current_count = await cache.get(rate_limit_key)

        if current_count and int(current_count) > 100:  # 100 requests per hour
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )

        # Generate case
        result = await case_generation_service.generate(
            request=request,
            user_id=current_user.sub
        )

        # Update rate limit
        await cache.incr(rate_limit_key)
        await cache.expire(rate_limit_key, 3600)  # 1 hour

        # Log activity
        await audit_service.log_activity(
            user_id=current_user.sub,
            action="generate_case",
            resource="case",
            resource_id=result.case_id,
            metadata={"grade": request.grade, "disorders": request.disorders}
        )

        return result

    except Exception as e:
        logger.error(f"Case generation failed: {e}")
        await audit_service.log_activity(
            user_id=current_user.sub,
            action="generate_case_failed",
            resource="case",
            metadata={"error": str(e)}
        )
        raise HTTPException(status_code=500, detail="Case generation failed")


@app.get("/api/cases", tags=["Case Management"])
async def list_cases(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(require_role([Role.CLINICIAN, Role.RESEARCHER, Role.ADMIN]))
):
    """
    List cases accessible to the user
    Researchers and admins see all cases, clinicians see only their own
    """
    try:
        if Role.ADMIN in current_user.roles or Role.RESEARCHER in current_user.roles:
            cases = await case_generation_service.list_all_cases(skip=skip, limit=limit)
        else:
            cases = await case_generation_service.list_user_cases(
                user_id=current_user.sub,
                skip=skip,
                limit=limit
            )

        await audit_service.log_activity(
            user_id=current_user.sub,
            action="list_cases",
            resource="cases"
        )

        return cases

    except Exception as e:
        logger.error(f"Failed to list cases: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cases")


@app.get("/api/cases/{case_id}", tags=["Case Management"])
async def get_case(
    case_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific case by ID"""
    try:
        case = await case_generation_service.get_case(case_id)

        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Check ownership or elevated permissions
        if (case.user_id != current_user.sub and
            Role.ADMIN not in current_user.roles and
            Role.RESEARCHER not in current_user.roles):
            raise HTTPException(status_code=403, detail="Access denied")

        await audit_service.log_activity(
            user_id=current_user.sub,
            action="view_case",
            resource="case",
            resource_id=case_id
        )

        return case

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get case: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve case")


@app.delete("/api/cases/{case_id}", tags=["Case Management"])
async def delete_case(
    case_id: str,
    current_user: User = Depends(require_role([Role.ADMIN]))
):
    """
    Delete a case
    Requires: admin role only
    """
    try:
        await case_generation_service.delete_case(case_id)

        await audit_service.log_activity(
            user_id=current_user.sub,
            action="delete_case",
            resource="case",
            resource_id=case_id
        )

        return {"status": "deleted", "case_id": case_id}

    except Exception as e:
        logger.error(f"Failed to delete case: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete case")


@app.get("/api/analytics/usage", tags=["Analytics"])
async def get_usage_analytics(
    days: int = 30,
    current_user: User = Depends(require_role([Role.RESEARCHER, Role.ADMIN]))
):
    """
    Get usage analytics
    Requires: researcher or admin role
    """
    try:
        analytics = await analytics_service.get_usage_stats(days=days)

        await audit_service.log_activity(
            user_id=current_user.sub,
            action="view_analytics",
            resource="analytics"
        )

        return analytics

    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@app.get("/api/analytics/cases", tags=["Analytics"])
async def get_case_analytics(
    current_user: User = Depends(require_role([Role.RESEARCHER, Role.ADMIN]))
):
    """
    Get case generation analytics
    Requires: researcher or admin role
    """
    try:
        analytics = await analytics_service.get_case_stats()

        return analytics

    except Exception as e:
        logger.error(f"Failed to get case analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics")


@app.post("/api/feedback", tags=["Feedback"])
async def submit_feedback(
    case_id: str,
    ratings: dict,
    category: str,
    detailed_feedback: str,
    current_user: User = Depends(get_current_user)
):
    """Submit feedback for a case"""
    try:
        feedback_id = await case_generation_service.save_feedback(
            case_id=case_id,
            user_id=current_user.sub,
            ratings=ratings,
            category=category,
            feedback=detailed_feedback
        )

        await audit_service.log_activity(
            user_id=current_user.sub,
            action="submit_feedback",
            resource="feedback",
            resource_id=feedback_id
        )

        return {"status": "success", "feedback_id": feedback_id}

    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@app.get("/api/audit/logs", tags=["Audit"])
async def get_audit_logs(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(require_role([Role.ADMIN]))
):
    """
    Get audit logs
    Requires: admin role only
    """
    try:
        logs = await audit_service.get_logs(skip=skip, limit=limit)
        return logs

    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit logs")


# User Management (Admin only)
@app.get("/api/users", tags=["User Management"])
async def list_users(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(require_role([Role.ADMIN]))
):
    """
    List all users
    Requires: admin role
    """
    # Implementation would integrate with Auth0 Management API
    pass


@app.put("/api/users/{user_id}/roles", tags=["User Management"])
async def update_user_roles(
    user_id: str,
    roles: List[str],
    current_user: User = Depends(require_role([Role.ADMIN]))
):
    """
    Update user roles
    Requires: admin role
    """
    # Implementation would integrate with Auth0 Management API
    await audit_service.log_activity(
        user_id=current_user.sub,
        action="update_user_roles",
        resource="user",
        resource_id=user_id,
        metadata={"new_roles": roles}
    )
    pass


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("ENVIRONMENT") == "development"
    )
