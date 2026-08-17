"""ClickHouse over its HTTP interface (:8123).

A native driver would buy nothing here: the offline jobs issue a handful of large aggregate
queries and are not latency-bound, and httpx is already a dependency for the article client.

The events table is `explained.user_events`, written by explAInedArticleEventConsumerService.
Note the database name is lowercase — ClickHouse identifiers are case-sensitive.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ClickHouseError(RuntimeError):
    """The query did not run. Callers treat this as an infrastructure failure (exit 2)."""


class ClickHouseClient:
    def __init__(
        self,
        base_url: str,
        database: str,
        user: str = "default",
        password: str = "",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._database = database
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout
        )
        self._params = {"database": database, "user": user}
        if password:
            self._params["password"] = password

    @property
    def database(self) -> str:
        return self._database

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def query_rows(
        self, sql: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a SELECT and return rows as dicts.

        `FORMAT JSONEachRow` is appended here so callers never think about serialization, and
        `parameters` are sent as ClickHouse server-side parameters (`param_x`) rather than
        interpolated into the SQL — the values come from CLI flags and must not be able to
        change the shape of the query.
        """
        payload = await self._post(f"{sql.rstrip().rstrip(';')} FORMAT JSONEachRow", parameters)

        rows: list[dict[str, Any]] = []
        for line in payload.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

        return rows

    async def execute(self, sql: str, parameters: dict[str, Any] | None = None) -> None:
        """Run a statement with no result set (DDL, INSERT ... FORMAT ...)."""
        await self._post(sql, parameters)

    async def _post(self, sql: str, parameters: dict[str, Any] | None = None) -> str:
        params = dict(self._params)
        for name, value in (parameters or {}).items():
            params[f"param_{name}"] = _render(value)

        try:
            response = await self._client.post("/", params=params, content=sql.encode("utf-8"))
        except httpx.HTTPError as exc:
            raise ClickHouseError(f"clickhouse request failed: {exc}") from exc

        if response.status_code != 200:
            raise ClickHouseError(
                f"clickhouse returned {response.status_code}: {response.text.strip()[:500]}"
            )

        return response.text


def _render(value: Any) -> str:
    """ClickHouse wants `DateTime64` parameters as `YYYY-MM-DD hh:mm:ss.mmm`, not ISO-8601."""
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    return str(value)
