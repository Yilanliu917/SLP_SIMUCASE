"""
Auth0 Authentication and RBAC Implementation
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
import os
import httpx
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# Auth0 Configuration
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_API_AUDIENCE = os.getenv("AUTH0_API_AUDIENCE")
AUTH0_ALGORITHMS = ["RS256"]

security = HTTPBearer()


class Role(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    CLINICIAN = "clinician"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Fine-grained permissions"""
    CREATE_CASE = "create:cases"
    READ_CASE = "read:cases"
    UPDATE_CASE = "update:cases"
    DELETE_CASE = "delete:cases"
    READ_ANALYTICS = "read:analytics"
    MANAGE_USERS = "manage:users"
    VIEW_AUDIT_LOGS = "view:audit_logs"
    EXPORT_DATA = "export:data"


class User(BaseModel):
    """User model from JWT token"""
    sub: str  # User ID from Auth0
    email: Optional[str] = None
    name: Optional[str] = None
    roles: List[Role] = []
    permissions: List[str] = []


class TokenPayload(BaseModel):
    """JWT Token payload"""
    sub: str
    exp: int
    iat: int
    aud: str
    permissions: Optional[List[str]] = []


# Cache for JWKS
_jwks_cache = None


async def get_jwks():
    """Fetch JSON Web Key Set from Auth0"""
    global _jwks_cache

    if _jwks_cache:
        return _jwks_cache

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://{AUTH0_DOMAIN}/.well-known/jwks.json")
            response.raise_for_status()
            _jwks_cache = response.json()
            return _jwks_cache
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to verify token"
        )


def get_signing_key(token: str, jwks: dict):
    """Extract signing key from JWKS"""
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to find appropriate key"
        )
    except Exception as e:
        logger.error(f"Error getting signing key: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenPayload:
    """
    Verify JWT token from Auth0
    """
    token = credentials.credentials

    try:
        # Get JWKS
        jwks = await get_jwks()
        signing_key = get_signing_key(token, jwks)

        # Verify and decode token
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=AUTH0_ALGORITHMS,
            audience=AUTH0_API_AUDIENCE,
            issuer=f"https://{AUTH0_DOMAIN}/"
        )

        return TokenPayload(**payload)

    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user(token_payload: TokenPayload = Depends(verify_token)) -> User:
    """
    Extract user information from verified token
    """
    try:
        # In production, you'd fetch additional user info from database
        # For now, we extract from token and Auth0 metadata

        user = User(
            sub=token_payload.sub,
            permissions=token_payload.permissions or []
        )

        # Extract roles from permissions or custom claims
        # Auth0 allows custom claims like "https://yourdomain.com/roles"
        if "admin" in user.permissions or "manage:users" in user.permissions:
            user.roles.append(Role.ADMIN)
        if "read:analytics" in user.permissions:
            user.roles.append(Role.RESEARCHER)
        if "create:cases" in user.permissions:
            user.roles.append(Role.CLINICIAN)

        # Default to viewer if no other roles
        if not user.roles:
            user.roles.append(Role.VIEWER)

        return user

    except Exception as e:
        logger.error(f"Failed to get current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


def require_role(allowed_roles: List[Role]):
    """
    Dependency to require specific roles
    Usage: current_user: User = Depends(require_role([Role.ADMIN, Role.RESEARCHER]))
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not any(role in current_user.roles for role in allowed_roles):
            logger.warning(
                f"User {current_user.sub} with roles {current_user.roles} "
                f"attempted to access endpoint requiring {allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[r.value for r in allowed_roles]}"
            )
        return current_user

    return role_checker


def require_permission(required_permission: Permission):
    """
    Dependency to require specific permission
    Usage: current_user: User = Depends(require_permission(Permission.DELETE_CASE))
    """
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if required_permission.value not in current_user.permissions:
            logger.warning(
                f"User {current_user.sub} attempted to access endpoint "
                f"requiring permission {required_permission.value}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_permission.value}"
            )
        return current_user

    return permission_checker


# Utility function for checking permissions programmatically
def has_permission(user: User, permission: Permission) -> bool:
    """Check if user has a specific permission"""
    return permission.value in user.permissions


def has_role(user: User, role: Role) -> bool:
    """Check if user has a specific role"""
    return role in user.roles
