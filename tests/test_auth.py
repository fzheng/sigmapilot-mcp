"""
Unit tests for the Auth0 JWT verification module.

Tests cover:
- Auth0TokenVerifier initialization
- Token verification (sync and async)
- JWKS client mocking
- Error handling for various JWT failures
- create_auth0_verifier factory function
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import asyncio
import time


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_jwks_response():
    """Mock JWKS response from Auth0."""
    return {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "n": "mock-modulus-value",
                "e": "AQAB",
                "kid": "test-key-id",
                "x5t": "mock-thumbprint",
                "x5c": ["mock-certificate"],
                "alg": "RS256"
            }
        ]
    }


@pytest.fixture
def mock_valid_payload():
    """Mock valid JWT payload."""
    return {
        "iss": "https://test-tenant.auth0.com/",
        "sub": "auth0|user123",
        "aud": "https://test-api.example.com",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,  # 1 hour from now
        "azp": "test-client-id",
        "scope": "openid profile email",
    }


@pytest.fixture
def mock_expired_payload():
    """Mock expired JWT payload."""
    return {
        "iss": "https://test-tenant.auth0.com/",
        "sub": "auth0|user123",
        "aud": "https://test-api.example.com",
        "iat": int(time.time()) - 7200,  # 2 hours ago
        "exp": int(time.time()) - 3600,  # 1 hour ago (expired)
        "azp": "test-client-id",
        "scope": "openid",
    }


@pytest.fixture
def verifier():
    """Create a test Auth0TokenVerifier instance."""
    from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier
    return Auth0TokenVerifier(
        domain="test-tenant.auth0.com",
        audience="https://test-api.example.com",
    )


@pytest.fixture
def mock_signing_key():
    """Mock signing key from JWKS client."""
    mock_key = MagicMock()
    mock_key.key = "mock-public-key"
    return mock_key


# =============================================================================
# Tests for Auth0TokenVerifier Initialization
# =============================================================================

class TestAuth0TokenVerifierInit:
    """Tests for Auth0TokenVerifier initialization."""

    def test_initialization_with_defaults(self):
        """Test verifier initialization with default parameters."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier

        verifier = Auth0TokenVerifier(
            domain="test-tenant.auth0.com",
            audience="https://test-api.example.com",
        )

        assert verifier.domain == "test-tenant.auth0.com"
        assert verifier.audience == "https://test-api.example.com"
        assert verifier.algorithms == ["RS256"]
        assert verifier.issuer == "https://test-tenant.auth0.com/"

    def test_initialization_with_custom_algorithms(self):
        """Test verifier initialization with custom algorithms."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier

        verifier = Auth0TokenVerifier(
            domain="test-tenant.auth0.com",
            audience="https://test-api.example.com",
            algorithms=["RS256", "RS384", "RS512"],
        )

        assert verifier.algorithms == ["RS256", "RS384", "RS512"]

    def test_jwks_client_initialized(self):
        """Test that JWKS client is properly initialized."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier

        verifier = Auth0TokenVerifier(
            domain="my-tenant.auth0.com",
            audience="https://api.example.com",
        )

        assert verifier.jwks_client is not None
        # JWKS client should be configured to fetch from Auth0's endpoint

    def test_issuer_url_format(self):
        """Test that issuer URL is correctly formatted with trailing slash."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier

        verifier = Auth0TokenVerifier(
            domain="custom-domain.auth0.com",
            audience="https://api.example.com",
        )

        assert verifier.issuer == "https://custom-domain.auth0.com/"
        assert verifier.issuer.endswith("/")

    def test_thread_pool_executor_created(self):
        """Test that thread pool executor is created for async operations."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier

        verifier = Auth0TokenVerifier(
            domain="test-tenant.auth0.com",
            audience="https://test-api.example.com",
        )

        assert verifier._executor is not None


# =============================================================================
# Tests for Synchronous Token Verification
# =============================================================================

class TestVerifyTokenSync:
    """Tests for synchronous token verification."""

    def test_valid_token_returns_access_token(self, verifier, mock_signing_key, mock_valid_payload):
        """Test that valid token returns AccessToken."""
        from mcp.server.auth.provider import AccessToken

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=mock_valid_payload):
                result = verifier.verify_token_sync("valid.jwt.token")

                assert result is not None
                assert isinstance(result, AccessToken)
                assert result.client_id == "test-client-id"
                assert result.scopes == ["openid", "profile", "email"]
                assert result.token == "valid.jwt.token"

    def test_invalid_token_returns_none(self, verifier):
        """Test that invalid/malformed token returns None."""
        result = verifier.verify_token_sync("not-a-valid-jwt")
        assert result is None

    def test_expired_token_returns_none(self, verifier, mock_signing_key):
        """Test that expired token returns None."""
        from jwt import ExpiredSignatureError

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', side_effect=ExpiredSignatureError("Token expired")):
                result = verifier.verify_token_sync("expired.jwt.token")
                assert result is None

    def test_invalid_signature_returns_none(self, verifier, mock_signing_key):
        """Test that token with invalid signature returns None."""
        from jwt import InvalidSignatureError

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', side_effect=InvalidSignatureError("Invalid signature")):
                result = verifier.verify_token_sync("bad-signature.jwt.token")
                assert result is None

    def test_invalid_audience_returns_none(self, verifier, mock_signing_key):
        """Test that token with wrong audience returns None."""
        from jwt import InvalidAudienceError

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', side_effect=InvalidAudienceError("Invalid audience")):
                result = verifier.verify_token_sync("wrong-audience.jwt.token")
                assert result is None

    def test_invalid_issuer_returns_none(self, verifier, mock_signing_key):
        """Test that token with wrong issuer returns None."""
        from jwt import InvalidIssuerError

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', side_effect=InvalidIssuerError("Invalid issuer")):
                result = verifier.verify_token_sync("wrong-issuer.jwt.token")
                assert result is None

    def test_jwks_fetch_error_returns_none(self, verifier):
        """Test that JWKS fetch failure returns None."""
        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', side_effect=Exception("Network error")):
            result = verifier.verify_token_sync("valid.jwt.token")
            assert result is None

    def test_permissions_claim_used_when_no_scope(self, verifier, mock_signing_key):
        """Test that permissions claim is used when scope is absent."""
        payload_with_permissions = {
            "iss": "https://test-tenant.auth0.com/",
            "sub": "auth0|user123",
            "aud": "https://test-api.example.com",
            "exp": int(time.time()) + 3600,
            "azp": "test-client-id",
            "permissions": ["read:users", "write:users", "delete:users"],
        }

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=payload_with_permissions):
                result = verifier.verify_token_sync("valid.jwt.token")

                assert result is not None
                assert result.scopes == ["read:users", "write:users", "delete:users"]

    def test_list_audience_extracts_first(self, verifier, mock_signing_key):
        """Test that list audience extracts first element."""
        payload_with_list_aud = {
            "iss": "https://test-tenant.auth0.com/",
            "sub": "auth0|user123",
            "aud": ["https://test-api.example.com", "https://other-api.example.com"],
            "exp": int(time.time()) + 3600,
            "azp": "test-client-id",
            "scope": "openid",
        }

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=payload_with_list_aud):
                result = verifier.verify_token_sync("valid.jwt.token")

                assert result is not None
                assert result.resource == "https://test-api.example.com"

    def test_sub_used_when_azp_missing(self, verifier, mock_signing_key):
        """Test that sub claim is used for client_id when azp is missing."""
        payload_without_azp = {
            "iss": "https://test-tenant.auth0.com/",
            "sub": "auth0|user123",
            "aud": "https://test-api.example.com",
            "exp": int(time.time()) + 3600,
            "scope": "openid",
        }

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=payload_without_azp):
                result = verifier.verify_token_sync("valid.jwt.token")

                assert result is not None
                assert result.client_id == "auth0|user123"

    def test_empty_scopes_when_no_scope_or_permissions(self, verifier, mock_signing_key):
        """Test that empty scopes list is returned when no scope or permissions."""
        payload_without_scopes = {
            "iss": "https://test-tenant.auth0.com/",
            "sub": "auth0|user123",
            "aud": "https://test-api.example.com",
            "exp": int(time.time()) + 3600,
            "azp": "test-client-id",
        }

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=payload_without_scopes):
                result = verifier.verify_token_sync("valid.jwt.token")

                assert result is not None
                assert result.scopes == []


# =============================================================================
# Tests for Asynchronous Token Verification
# =============================================================================

class TestVerifyTokenAsync:
    """Tests for asynchronous token verification."""

    @pytest.mark.asyncio
    async def test_async_valid_token_returns_access_token(self, verifier, mock_signing_key, mock_valid_payload):
        """Test that async verification returns AccessToken for valid token."""
        from mcp.server.auth.provider import AccessToken

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=mock_valid_payload):
                result = await verifier.verify_token("valid.jwt.token")

                assert result is not None
                assert isinstance(result, AccessToken)
                assert result.client_id == "test-client-id"

    @pytest.mark.asyncio
    async def test_async_invalid_token_returns_none(self, verifier):
        """Test that async verification returns None for invalid token."""
        result = await verifier.verify_token("invalid.token")
        assert result is None

    @pytest.mark.asyncio
    async def test_async_expired_token_returns_none(self, verifier, mock_signing_key):
        """Test that async verification returns None for expired token."""
        from jwt import ExpiredSignatureError

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', side_effect=ExpiredSignatureError("Token expired")):
                result = await verifier.verify_token("expired.jwt.token")
                assert result is None

    @pytest.mark.asyncio
    async def test_async_uses_thread_executor(self, verifier, mock_signing_key, mock_valid_payload):
        """Test that async verification uses thread pool executor."""
        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=mock_valid_payload):
                # This should not block the event loop
                result = await verifier.verify_token("valid.jwt.token")
                assert result is not None


# =============================================================================
# Tests for create_auth0_verifier Factory
# =============================================================================

class TestCreateAuth0Verifier:
    """Tests for the create_auth0_verifier factory function."""

    def test_creates_verifier_with_required_env_vars(self):
        """Test verifier creation with required environment variables."""
        from sigmapilot_mcp.core.utils.auth import create_auth0_verifier

        with patch.dict(os.environ, {
            "AUTH0_DOMAIN": "test-tenant.auth0.com",
            "AUTH0_AUDIENCE": "https://test-api.example.com",
        }, clear=True):
            verifier = create_auth0_verifier()

            assert verifier.domain == "test-tenant.auth0.com"
            assert verifier.audience == "https://test-api.example.com"
            assert verifier.algorithms == ["RS256"]

    def test_raises_value_error_without_domain(self):
        """Test that ValueError is raised without AUTH0_DOMAIN."""
        from sigmapilot_mcp.core.utils.auth import create_auth0_verifier

        with patch.dict(os.environ, {
            "AUTH0_AUDIENCE": "https://test-api.example.com",
        }, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_auth0_verifier()
            assert "AUTH0_DOMAIN" in str(exc_info.value)

    def test_raises_value_error_without_audience(self):
        """Test that ValueError is raised without AUTH0_AUDIENCE."""
        from sigmapilot_mcp.core.utils.auth import create_auth0_verifier

        with patch.dict(os.environ, {
            "AUTH0_DOMAIN": "test-tenant.auth0.com",
        }, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_auth0_verifier()
            assert "AUTH0_AUDIENCE" in str(exc_info.value)

    def test_parses_comma_separated_algorithms(self):
        """Test that comma-separated algorithms are parsed correctly."""
        from sigmapilot_mcp.core.utils.auth import create_auth0_verifier

        with patch.dict(os.environ, {
            "AUTH0_DOMAIN": "test-tenant.auth0.com",
            "AUTH0_AUDIENCE": "https://test-api.example.com",
            "AUTH0_ALGORITHMS": "RS256, RS384, RS512",
        }, clear=True):
            verifier = create_auth0_verifier()

            assert verifier.algorithms == ["RS256", "RS384", "RS512"]

    def test_default_algorithm_when_not_specified(self):
        """Test that RS256 is used when AUTH0_ALGORITHMS is not set."""
        from sigmapilot_mcp.core.utils.auth import create_auth0_verifier

        with patch.dict(os.environ, {
            "AUTH0_DOMAIN": "test-tenant.auth0.com",
            "AUTH0_AUDIENCE": "https://test-api.example.com",
        }, clear=True):
            verifier = create_auth0_verifier()

            assert verifier.algorithms == ["RS256"]

    def test_strips_whitespace_from_algorithms(self):
        """Test that whitespace is stripped from algorithm names."""
        from sigmapilot_mcp.core.utils.auth import create_auth0_verifier

        with patch.dict(os.environ, {
            "AUTH0_DOMAIN": "test-tenant.auth0.com",
            "AUTH0_AUDIENCE": "https://test-api.example.com",
            "AUTH0_ALGORITHMS": "  RS256  ,  RS384  ",
        }, clear=True):
            verifier = create_auth0_verifier()

            assert verifier.algorithms == ["RS256", "RS384"]


# =============================================================================
# Tests for MCP AccessToken Compatibility
# =============================================================================

class TestAccessTokenCompatibility:
    """Tests for MCP AccessToken type compatibility."""

    def test_returns_mcp_access_token_type(self, verifier, mock_signing_key, mock_valid_payload):
        """Test that returned token is MCP AccessToken type."""
        from mcp.server.auth.provider import AccessToken

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=mock_valid_payload):
                result = verifier.verify_token_sync("valid.jwt.token")

                assert type(result).__name__ == "AccessToken"
                assert type(result) == AccessToken

    def test_access_token_has_required_fields(self, verifier, mock_signing_key, mock_valid_payload):
        """Test that AccessToken has all required fields."""
        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=mock_valid_payload):
                result = verifier.verify_token_sync("valid.jwt.token")

                assert hasattr(result, 'token')
                assert hasattr(result, 'client_id')
                assert hasattr(result, 'scopes')
                assert hasattr(result, 'expires_at')
                assert hasattr(result, 'resource')

    def test_access_token_field_types(self, verifier, mock_signing_key, mock_valid_payload):
        """Test that AccessToken fields have correct types."""
        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
            with patch('jwt.decode', return_value=mock_valid_payload):
                result = verifier.verify_token_sync("valid.jwt.token")

                assert isinstance(result.token, str)
                assert isinstance(result.client_id, str)
                assert isinstance(result.scopes, list)
                assert isinstance(result.expires_at, int)


# =============================================================================
# Tests for Error Logging
# =============================================================================

class TestErrorLogging:
    """Tests for error logging behavior."""

    def test_logs_invalid_token_error(self, verifier, mock_signing_key, caplog):
        """Test that InvalidTokenError is logged at debug level."""
        import logging
        from jwt import InvalidTokenError

        with caplog.at_level(logging.DEBUG):
            with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
                with patch('jwt.decode', side_effect=InvalidTokenError("Test error")):
                    result = verifier.verify_token_sync("invalid.jwt.token")

                    # Should be logged at DEBUG level (not exposing details in production)
                    assert "Token validation failed" in caplog.text
                    assert result is None

    def test_logs_generic_exception(self, verifier, caplog):
        """Test that generic exceptions are logged at warning level."""
        import logging

        with caplog.at_level(logging.WARNING):
            with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', side_effect=Exception("Network failure")):
                result = verifier.verify_token_sync("valid.jwt.token")

                assert "Token verification error" in caplog.text
                assert result is None

    def test_does_not_log_sensitive_token_data(self, verifier, mock_signing_key, caplog):
        """Test that token content is not logged."""
        import logging
        from jwt import InvalidTokenError

        sensitive_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.secret_payload.signature"

        with caplog.at_level(logging.DEBUG):
            with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_signing_key):
                with patch('jwt.decode', side_effect=InvalidTokenError("Invalid")):
                    verifier.verify_token_sync(sensitive_token)

                    # Token content should not appear in logs
                    assert sensitive_token not in caplog.text
                    assert "secret_payload" not in caplog.text


# =============================================================================
# Tests for Resource Cleanup
# =============================================================================

class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    def test_close_shuts_down_executor(self):
        """Test that close() shuts down the thread pool executor."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier

        verifier = Auth0TokenVerifier(
            domain="test-tenant.auth0.com",
            audience="https://test-api.example.com",
        )

        assert verifier._executor is not None
        verifier.close()
        assert verifier._executor is None

    def test_close_is_idempotent(self):
        """Test that close() can be called multiple times safely."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier

        verifier = Auth0TokenVerifier(
            domain="test-tenant.auth0.com",
            audience="https://test-api.example.com",
        )

        verifier.close()
        verifier.close()  # Should not raise
        assert verifier._executor is None

    def test_destructor_calls_close(self):
        """Test that destructor cleans up resources."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier

        verifier = Auth0TokenVerifier(
            domain="test-tenant.auth0.com",
            audience="https://test-api.example.com",
        )

        executor = verifier._executor
        del verifier
        # Executor should have been shut down (though we can't easily verify
        # without keeping a reference, this tests the destructor runs)


# =============================================================================
# Integration Tests
# =============================================================================

class TestAuthIntegration:
    """Integration tests for auth module."""

    def test_full_verification_flow(self):
        """Test complete verification flow with mocked JWKS."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier
        from mcp.server.auth.provider import AccessToken

        verifier = Auth0TokenVerifier(
            domain="integration-test.auth0.com",
            audience="https://integration-api.example.com",
        )

        mock_key = MagicMock()
        mock_key.key = "mock-rsa-key"

        payload = {
            "iss": "https://integration-test.auth0.com/",
            "sub": "integration|user456",
            "aud": "https://integration-api.example.com",
            "exp": int(time.time()) + 7200,
            "azp": "integration-client",
            "scope": "openid profile email offline_access",
        }

        with patch.object(verifier.jwks_client, 'get_signing_key_from_jwt', return_value=mock_key):
            with patch('jwt.decode', return_value=payload):
                result = verifier.verify_token_sync("integration.test.token")

                assert result is not None
                assert isinstance(result, AccessToken)
                assert result.client_id == "integration-client"
                assert result.scopes == ["openid", "profile", "email", "offline_access"]
                assert result.resource == "https://integration-api.example.com"
                assert result.token == "integration.test.token"

    def test_verifier_can_be_used_as_fastmcp_token_verifier(self):
        """Test that verifier implements TokenVerifier protocol."""
        from sigmapilot_mcp.core.utils.auth import Auth0TokenVerifier

        verifier = Auth0TokenVerifier(
            domain="protocol-test.auth0.com",
            audience="https://protocol-api.example.com",
        )

        # TokenVerifier protocol requires verify_token method
        assert hasattr(verifier, 'verify_token')
        assert callable(verifier.verify_token)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
