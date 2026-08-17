"""Identity-service client, used only by the dev seeding scripts.

The user id we need everywhere else is the JWT `sub` claim — the same value the .NET services
read (`MapInboundClaims = false`, `JwtRegisteredClaimNames.Sub`). We decode it without
verifying the signature on purpose: we are not authenticating anything here, we already hold
the token the server just issued, and verifying would mean shipping the signing key and a JWT
library to a fixture script.
"""

import base64
import json
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


class IdentityError(RuntimeError):
    """Registration or login did not succeed."""


@dataclass(frozen=True, slots=True)
class SeededUser:
    user_id: str
    email: str
    nickname: str
    access_token: str


class IdentityClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def ensure_user(self, email: str, nickname: str, password: str) -> SeededUser:
        """Register then log in. A second run re-uses the existing account."""
        await self._register(email, nickname, password)
        token = await self.login(email, password)

        return SeededUser(user_id=subject_of(token), email=email, nickname=nickname, access_token=token)

    async def _register(self, email: str, nickname: str, password: str) -> None:
        try:
            response = await self._client.post(
                "/user", json={"email": email, "nickname": nickname, "password": password}
            )
        except httpx.HTTPError as exc:
            raise IdentityError(f"POST /user failed: {exc}") from exc

        # 400 is the "already registered" answer, which is the normal case on a rerun.
        if response.status_code not in (200, 201, 400, 409):
            raise IdentityError(f"POST /user returned {response.status_code}: {response.text[:200]}")

    async def login(self, email: str, password: str) -> str:
        try:
            response = await self._client.post(
                "/session", json={"email": email, "password": password, "rememberMe": False}
            )
        except httpx.HTTPError as exc:
            raise IdentityError(f"POST /session failed: {exc}") from exc

        if response.status_code != 200:
            raise IdentityError(f"POST /session returned {response.status_code}: {response.text[:200]}")

        token = response.json().get("accessToken")
        if not token:
            raise IdentityError("login succeeded but returned no accessToken")

        return str(token)


def subject_of(token: str) -> str:
    """The `sub` claim, without signature verification (see the module docstring)."""
    parts = token.split(".")
    if len(parts) != 3:
        raise IdentityError("access token is not a three-segment JWT")

    payload = parts[1]
    padding = "=" * (-len(payload) % 4)

    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + padding))
    except (ValueError, json.JSONDecodeError) as exc:
        raise IdentityError(f"could not decode the JWT payload: {exc}") from exc

    subject = claims.get("sub")
    if not subject:
        raise IdentityError("access token carries no `sub` claim")

    return str(subject)
