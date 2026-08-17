"""Article-service client.

The repository decided against reading another service's PostgreSQL directly
(explAInedRecommendationService/CONTRACT.md, section 0 of todo.md), and that holds offline too.

There is no "list every article" endpoint, so the catalog is walked through
`GET /articles/recent?limit=100&offset=N`. That endpoint serves exactly Published + Public,
which is exactly the set the feed can ever show — embedding anything else would be wasted work.

Caveat inherited from the article service: `PublishedAt` is stamped at creation and never moves
on publish, so `recent` is really "recently created" and an edit to an old article does not
resurface. `--full` on the embeddings job is the answer to that.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)

RECENT_PAGE_SIZE = 100
BATCH_MAX_IDS = 100


class ArticleServiceError(RuntimeError):
    """The article service did not answer. Infrastructure failure, not an empty catalog."""


@dataclass(frozen=True, slots=True)
class Article:
    id: str
    title: str
    content: str
    description: str
    tags: str
    author_id: str
    published_at: datetime | None
    updated_at: datetime | None
    word_count: int

    @property
    def embedding_text(self) -> str:
        """Title first: it carries the most topical signal per token, and MiniLM truncates."""
        parts = [self.title, self.description, self.content]
        return "\n".join(p.strip() for p in parts if p and p.strip())

    @property
    def tag_list(self) -> list[str]:
        """`Tags` is one delimited string in the article model — there is no tags table.

        Kept here rather than in feature_store because it is a property of the wire format,
        not of any feature. Tag-based features themselves are deferred (see CONTRACT.md).
        """
        raw = self.tags.replace(";", ",").replace("|", ",")
        return [t.strip().lower() for t in raw.split(",") if t.strip()]


class ArticleClient:
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

    async def iter_published(self, page_size: int = RECENT_PAGE_SIZE) -> AsyncIterator[Article]:
        """Walk the whole Published + Public catalog, newest first."""
        offset = 0

        while True:
            page = await self.recent(limit=page_size, offset=offset)
            if not page:
                return

            for article in page:
                yield article

            if len(page) < page_size:
                return

            offset += len(page)

    async def recent(self, limit: int = RECENT_PAGE_SIZE, offset: int = 0) -> list[Article]:
        payload = await self._get("/articles/recent", params={"limit": limit, "offset": offset})
        return [_parse(row) for row in payload]

    async def get_batch(self, article_ids: list[str]) -> list[Article]:
        """`/articles/batch` caps at 100 ids and drops unknown ones, so the result may be shorter."""
        if not article_ids:
            return []

        out: list[Article] = []
        for start in range(0, len(article_ids), BATCH_MAX_IDS):
            chunk = article_ids[start : start + BATCH_MAX_IDS]
            payload = await self._get("/articles/batch", params={"ids": ",".join(chunk)})
            out.extend(_parse(row) for row in payload)

        return out

    async def _get(self, path: str, params: dict[str, object]) -> list[dict]:
        try:
            response = await self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise ArticleServiceError(f"GET {path} failed: {exc}") from exc

        if response.status_code != 200:
            raise ArticleServiceError(f"GET {path} returned {response.status_code}")

        body = response.json()
        if not isinstance(body, list):
            raise ArticleServiceError(f"GET {path} returned {type(body).__name__}, expected a list")

        return body


def _parse(row: dict) -> Article:
    return Article(
        id=str(row.get("id") or row.get("Id") or ""),
        title=_text(row, "title"),
        content=_text(row, "content"),
        description=_text(row, "description"),
        tags=_text(row, "tags"),
        author_id=_text(row, "authorId"),
        published_at=_timestamp(row, "publishedAt"),
        updated_at=_timestamp(row, "updatedAt"),
        word_count=int(row.get("wordCount") or row.get("WordCount") or 0),
    )


def _text(row: dict, camel: str) -> str:
    value = row.get(camel)
    if value is None:
        value = row.get(camel[0].upper() + camel[1:])

    return str(value) if value is not None else ""


def _timestamp(row: dict, camel: str) -> datetime | None:
    raw = _text(row, camel)
    if not raw:
        return None

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("unparseable %s: %r", camel, raw)
        return None
