"""
Auth0 JWT Token Verification Utilities.

This module provides JWT token verification for Auth0 authentication.
It validates tokens using Auth0's JWKS (JSON Web Key Set) endpoint.

The Auth0TokenVerifier implements the MCP TokenVerifier protocol,
allowing it to be used directly with FastMCP for authentication.
"""

from __future__ import annotations

import logging
import os
import asyncio
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

# MCP AccessToken for compatibility with FastMCP
from mcp.server.auth.provider import AccessToken

# Configure module logger
logger = logging.getLogger(__name__)

# JWT verification dependencies
try:
    import jwt
    from jwt import PyJWKClient, InvalidTokenError
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


class Auth0TokenVerifier:
    """
    Verifies Auth0 JWT tokens using JWKS.

    This verifier fetches the signing key from Auth0's JWKS endpoint
    and validates the JWT signature, expiration, audience, and issuer.
    """

    def __init__(
        self,
        domain: str,
        audience: str,
        algorithms: List[str] = None,
    ):
        """
        Initialize the Auth0 token verifier.

        Args:
            domain: Auth0 domain (e.g., 'your-tenant.auth0.com')
            audience: API identifier configured in Auth0
            algorithms: JWT signing algorithms (default: ['RS256'])
        """
        if not JWT_AVAILABLE:
            raise ImportError(
                "PyJWT is required for Auth0 authentication. "
                "Install with: pip install PyJWT"
            )

        self.domain = domain
        self.audience = audience
        self.algorithms = algorithms or ["RS256"]
        self.issuer = f"https://{domain}/"

        # JWKS endpoint for fetching public keys
        jwks_url = f"https://{domain}/.well-known/jwks.json"
        self.jwks_client = PyJWKClient(jwks_url)

        # Thread pool for async key fetching
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        """
        Verify a JWT token and return the validated claims.

        Args:
            token: The JWT token string to verify

        Returns:
            AccessToken if valid, None if invalid
        """
        try:
            # Get the signing key asynchronously
            loop = asyncio.get_event_loop()
            signing_key = await loop.run_in_executor(
                self._executor,
                self.jwks_client.get_signing_key_from_jwt,
                token
            )

            # Decode and validate the JWT
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                }
            )

            # Extract scopes from 'scope' claim (space-separated) or 'permissions' claim
            scopes = []
            if "scope" in payload:
                scopes = payload["scope"].split()
            elif "permissions" in payload:
                scopes = payload["permissions"]

            # Handle audience - can be string or list
            aud = payload.get("aud")
            resource = aud[0] if isinstance(aud, list) else aud

            return AccessToken(
                token=token,
                client_id=payload.get("azp", payload.get("sub", "")),
                scopes=scopes,
                expires_at=payload.get("exp"),
                resource=resource,
            )

        except InvalidTokenError as e:
            # Token validation failed (expired, wrong signature, etc.)
            # Log at debug level to avoid exposing token details in production logs
            logger.debug(f"Token validation failed: {type(e).__name__}")
            return None
        except Exception as e:
            # Unexpected error during verification
            logger.warning(f"Token verification error: {type(e).__name__}")
            return None

    def verify_token_sync(self, token: str) -> Optional[AccessToken]:
        """
        Synchronous version of verify_token.

        Args:
            token: The JWT token string to verify

        Returns:
            AccessToken if valid, None if invalid
        """
        try:
            # Get the signing key
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)

            # Decode and validate the JWT
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                }
            )

            # Extract scopes
            scopes = []
            if "scope" in payload:
                scopes = payload["scope"].split()
            elif "permissions" in payload:
                scopes = payload["permissions"]

            # Handle audience - can be string or list
            aud = payload.get("aud")
            resource = aud[0] if isinstance(aud, list) else aud

            return AccessToken(
                token=token,
                client_id=payload.get("azp", payload.get("sub", "")),
                scopes=scopes,
                expires_at=payload.get("exp"),
                resource=resource,
            )

        except InvalidTokenError as e:
            # Log at debug level to avoid exposing token details in production logs
            logger.debug(f"Token validation failed: {type(e).__name__}")
            return None
        except Exception as e:
            logger.warning(f"Token verification error: {type(e).__name__}")
            return None

    def close(self) -> None:
        """
        Clean up resources used by the verifier.

        Should be called when the verifier is no longer needed to prevent
        resource leaks from the thread pool executor.
        """
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __del__(self):
        """Destructor to ensure executor cleanup."""
        self.close()


def create_auth0_verifier() -> Auth0TokenVerifier:
    """
    Create an Auth0 token verifier from environment variables.

    Required environment variables:
        AUTH0_DOMAIN: Your Auth0 domain (e.g., 'your-tenant.auth0.com')
        AUTH0_AUDIENCE: API identifier from Auth0

    Optional environment variables:
        AUTH0_ALGORITHMS: Comma-separated list of algorithms (default: 'RS256')

    Returns:
        Configured Auth0TokenVerifier instance

    Raises:
        ValueError: If required environment variables are missing
    """
    domain = os.getenv("AUTH0_DOMAIN")
    audience = os.getenv("AUTH0_AUDIENCE")

    if not domain:
        raise ValueError("AUTH0_DOMAIN environment variable is required")
    if not audience:
        raise ValueError("AUTH0_AUDIENCE environment variable is required")

    # Parse algorithms (default to RS256)
    algorithms_str = os.getenv("AUTH0_ALGORITHMS", "RS256")
    algorithms = [a.strip() for a in algorithms_str.split(",")]

    return Auth0TokenVerifier(
        domain=domain,
        audience=audience,
        algorithms=algorithms,
    )
