"""Create a usable development corpus through the public APIs.

    python -m explained_ml.scripts.seed_dev_corpus --articles 200 --users 12

The trap this script exists to work around: `POST /articles` hardcodes `Status = Draft` and
ignores whatever status you send (`ArticleService.SaveArticleAsync`). A one-call seeder adds
Drafts, which `/articles/recent` and `/articles/batch` both filter out — so the article service
keeps reporting an empty catalog and every downstream job keeps exiting 1. Publishing needs a
second call, `PUT /articles/{id}` with `status: "published"`.

Writes `seed_users.json` next to the working directory so `simulate_behavior` can reuse the
same accounts instead of re-deriving them.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx

from ..config import get_settings
from ..identity import IdentityClient, IdentityError, SeededUser
from ..logs import EXIT_INFRA, EXIT_OK, configure
from .corpus import generate

logger = logging.getLogger("explained_ml.seed_dev_corpus")

DEFAULT_PASSWORD = "Password1"
USERS_FILE = Path("seed_users.json")


async def seed(
    articles: int,
    users: int,
    seed_value: int,
    identity_url: str,
    articles_url: str,
    users_file: Path,
    profiles_url: str | None = None,
) -> int:
    identity = IdentityClient(identity_url)
    http = httpx.AsyncClient(base_url=articles_url.rstrip("/"), timeout=30.0)
    profiles = (
        httpx.AsyncClient(base_url=profiles_url.rstrip("/"), timeout=30.0)
        if profiles_url
        else None
    )

    try:
        accounts = await _ensure_users(identity, users)
        if not accounts:
            logger.error("no accounts could be created — is the identity service on %s?", identity_url)
            return EXIT_INFRA

        logger.info("seeded %d accounts", len(accounts))

        if profiles is not None:
            named = await _name_profiles(profiles, accounts)
            logger.info("named %d of %d profiles", named, len(accounts))

        published = await _publish_articles(http, accounts, generate(articles, seed_value))
        logger.info("published %d of %d articles", published, articles)

        users_file.write_text(
            json.dumps(
                [
                    {"user_id": a.user_id, "email": a.email, "nickname": a.nickname}
                    for a in accounts
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s", users_file)

        if published == 0:
            logger.error("nothing was published; the corpus is unchanged")
            return EXIT_INFRA

        return EXIT_OK
    except (IdentityError, httpx.HTTPError) as exc:
        logger.error("seeding failed: %s", exc)
        return EXIT_INFRA
    finally:
        await identity.close()
        await http.aclose()
        if profiles is not None:
            await profiles.aclose()


async def _ensure_users(identity: IdentityClient, count: int) -> list[SeededUser]:
    accounts: list[SeededUser] = []

    for index in range(count):
        email = f"seed{index:02d}@explained.local"
        try:
            accounts.append(await identity.ensure_user(email, f"seed{index:02d}", DEFAULT_PASSWORD))
        except IdentityError as exc:
            logger.warning("could not seed %s: %s", email, exc)

    return accounts


async def _publish_articles(
    http: httpx.AsyncClient, accounts: list[SeededUser], drafts: list
) -> int:
    published = 0

    for index, draft in enumerate(drafts):
        author = accounts[index % len(accounts)]
        headers = {"Authorization": f"Bearer {author.access_token}"}

        created = await http.post(
            "/articles",
            headers=headers,
            json={
                "title": draft.title,
                "content": draft.content,
                "description": draft.description,
                "tags": draft.tags,
                "accessLevel": "public",
            },
        )

        if created.status_code not in (200, 201):
            logger.warning("create failed for %r: %s", draft.title, created.status_code)
            continue

        article_id = created.json().get("id") or created.json().get("Id")
        if not article_id:
            logger.warning("create returned no id for %r", draft.title)
            continue

        # The second call is not optional — see the module docstring.
        updated = await http.put(
            f"/articles/{article_id}",
            headers=headers,
            json={
                "title": draft.title,
                "content": draft.content,
                "description": draft.description,
                "tags": draft.tags,
                "accessLevel": "public",
                "status": "published",
            },
        )

        if updated.status_code not in (200, 204):
            logger.warning("publish failed for %s: %s", article_id, updated.status_code)
            continue

        published += 1

    return published


async def _name_profiles(profiles: httpx.AsyncClient, accounts: list) -> int:
    named = 0

    for account in accounts:
        headers = {"Authorization": f"Bearer {account.access_token}"}

        try:
            response = await profiles.put(
                "/profiles/me",
                headers=headers,
                json={"displayName": account.nickname, "bio": f"seeded account {account.nickname}"},
            )
        except httpx.HTTPError as exc:
            logger.warning("profile service unreachable, leaving names unset: %s", exc)
            return named

        if response.status_code in (200, 204):
            named += 1
        else:
            logger.warning(
                "PUT /profiles/me for %s returned %s", account.email, response.status_code
            )

    return named


def cli() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Seed a topically structured dev corpus")
    parser.add_argument("--articles", type=int, default=200)
    parser.add_argument("--users", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity-url", default=settings.identity_base_url)
    parser.add_argument("--articles-url", default=settings.articles_base_url)
    parser.add_argument("--profiles-url", default=settings.profiles_base_url)
    parser.add_argument(
        "--no-profiles",
        action="store_true",
        help="skip naming profiles (use when the profile service is not running)",
    )
    parser.add_argument("--users-file", type=Path, default=USERS_FILE)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure(args.log_level)

    return asyncio.run(
        seed(
            args.articles,
            args.users,
            args.seed,
            args.identity_url,
            args.articles_url,
            args.users_file,
            profiles_url=None if args.no_profiles else args.profiles_url,
        )
    )


if __name__ == "__main__":
    sys.exit(cli())
