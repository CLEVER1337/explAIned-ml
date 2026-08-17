"""Article text -> `rec:article_embedding:{aid}`, hourly.

    python -m explained_ml.jobs.embeddings            # incremental
    python -m explained_ml.jobs.embeddings --full     # re-encode everything

The catalog is walked over HTTP (`GET /articles/recent`), not read from the article service's
PostgreSQL — see articles.py for why. That endpoint serves exactly Published + Public, which is
exactly the set the feed can show, so nothing useful is missed.

Incremental mode leans on a quirk: `/articles/recent` orders by `PublishedAt DESC` and the
article service stamps `PublishedAt` at *creation*, so genuinely new articles are always on the
first page. Once a whole page is already-known content, everything after it is older still, and
we stop. What this misses is an *edit* to an old article, which is what `--full` is for.

Vectors are written L2-normalised. explAIned-faiss normalises again at build time, so this is
belt and braces — but a normalised vector is also what `x_embedding_cosine` assumes, and that
feature reads Redis directly without going through faiss.
"""

import argparse
import asyncio
import logging
import sys
from contextlib import aclosing

import numpy as np

from ..articles import Article, ArticleClient, ArticleServiceError
from ..codec import normalize
from ..config import Settings, get_settings
from ..encoders import TextEncoder, build_encoder, fingerprint
from ..logs import EXIT_INFRA, EXIT_NOTHING_TO_DO, EXIT_OK, configure
from ..redis_store import FINGERPRINTS_KEY, RecommendationStore

logger = logging.getLogger("explained_ml.embeddings")


async def build(
    settings: Settings,
    encoder: TextEncoder,
    full: bool = False,
    stop_after_clean_pages: int = 2,
    batch_size: int = 32,
    max_articles: int | None = None,
    articles: ArticleClient | None = None,
    store: RecommendationStore | None = None,
) -> int:
    owns = articles is None and store is None
    articles = articles or ArticleClient(settings.articles_base_url, settings.request_timeout_seconds)
    store = store or RecommendationStore(settings.redis_url, settings.embedding_dim)

    try:
        known = {} if full else await store.get_fingerprints(FINGERPRINTS_KEY)
        logger.info("%d articles already embedded", len(known))

        pending: list[Article] = []
        seen = 0
        clean_pages = 0
        page: list[Article] = []

        # aclosing because the common case is leaving this loop early; an abandoned async
        # generator is finalised by the event loop whenever it feels like it, which surfaces as
        # a ResourceWarning on an otherwise correct path and hides the ones that matter.
        async with aclosing(articles.iter_published()) as catalog:
            async for article in catalog:
                seen += 1
                page.append(article)

                if len(page) >= 100:
                    clean_pages = _account(page, known, pending, clean_pages)
                    page = []

                    if not full and clean_pages >= stop_after_clean_pages:
                        logger.info("%d consecutive unchanged pages, stopping the walk", clean_pages)
                        break

                if max_articles is not None and seen >= max_articles:
                    break

        if page:
            _account(page, known, pending, clean_pages)

        if not pending:
            logger.info("nothing to encode: %d articles seen, all fingerprints current", seen)
            return EXIT_NOTHING_TO_DO

        logger.info("encoding %d of %d articles seen", len(pending), seen)
        encoded = _encode(pending, encoder, settings.embedding_dim, batch_size)

        await store.set_article_embeddings(encoded)
        await store.set_fingerprints(
            FINGERPRINTS_KEY, {a.id: fingerprint(a.embedding_text) for a in pending}
        )
        await store.mark_fresh("article_embedding")

        logger.info("wrote %d embeddings", len(encoded))
        return EXIT_OK
    except ArticleServiceError as exc:
        logger.error("catalog walk failed, existing embeddings untouched: %s", exc)
        return EXIT_INFRA
    finally:
        if owns:
            await articles.close()
            await store.close()


def _account(
    page: list[Article], known: dict[str, str], pending: list[Article], clean_pages: int
) -> int:
    changed = [a for a in page if known.get(a.id) != fingerprint(a.embedding_text)]
    pending.extend(changed)

    return clean_pages + 1 if not changed else 0


def _encode(
    articles: list[Article], encoder: TextEncoder, dim: int, batch_size: int
) -> dict[str, np.ndarray]:
    encoded: dict[str, np.ndarray] = {}

    for start in range(0, len(articles), batch_size):
        chunk = articles[start : start + batch_size]
        vectors = encoder.encode([a.embedding_text for a in chunk])

        for article, vector in zip(chunk, vectors, strict=True):
            if vector.shape != (dim,):
                logger.warning("skipping %s: encoder returned shape %s", article.id, vector.shape)
                continue

            encoded[article.id] = normalize(vector)

    return encoded


def cli() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Embed published articles into Redis")
    parser.add_argument("--full", action="store_true", help="re-encode the whole catalog")
    parser.add_argument("--stop-after-clean-pages", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument(
        "--encoder",
        choices=("sbert", "hashing"),
        default="sbert",
        help="hashing is a dev fixture producing meaningless vectors; see encoders.py",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure(args.log_level)

    try:
        encoder = build_encoder(
            args.encoder, settings.embedding_model, settings.embedding_dim, args.batch_size
        )
    except (ImportError, ValueError, OSError) as exc:
        logger.error("could not load the encoder: %s", exc)
        return EXIT_INFRA

    return asyncio.run(
        build(
            settings,
            encoder,
            full=args.full,
            stop_after_clean_pages=args.stop_after_clean_pages,
            batch_size=args.batch_size,
            max_articles=args.max_articles,
        )
    )


if __name__ == "__main__":
    sys.exit(cli())
