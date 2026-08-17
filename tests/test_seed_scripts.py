import base64
import json
import random
from datetime import UTC, datetime

import pytest

from explained_ml.articles import Article
from explained_ml.identity import IdentityError, subject_of
from explained_ml.scripts.corpus import TOPIC_NAMES, generate
from explained_ml.scripts.simulate_behavior import plan_events, topic_of

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)


def token_with(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def article(article_id: str, tags: str) -> Article:
    return Article(
        id=article_id,
        title="t",
        content="c",
        description="d",
        tags=tags,
        author_id="author",
        published_at=NOW,
        updated_at=NOW,
        word_count=10,
    )


# --- corpus ---------------------------------------------------------------------------


def test_corpus_is_deterministic_for_a_seed():
    assert generate(20, seed=1) == generate(20, seed=1)
    assert generate(20, seed=1) != generate(20, seed=2)


def test_corpus_spreads_across_every_topic():
    topics = {a.topic for a in generate(len(TOPIC_NAMES) * 2, seed=3)}

    assert topics == set(TOPIC_NAMES)


def test_articles_are_long_enough_for_a_sentence_encoder_to_chew_on():
    # The live corpus has 4-character bodies; embedding those produces noise, which is the
    # entire reason this generator exists.
    for draft in generate(10, seed=4):
        assert len(draft.content.split()) > 60
        assert "\n\n" in draft.content


def test_tags_come_from_the_article_topic():
    from explained_ml.scripts.corpus import TOPICS

    for draft in generate(10, seed=5):
        assert all(tag.strip() in TOPICS[draft.topic]["tags"] for tag in draft.tags.split(","))


# --- identity -------------------------------------------------------------------------


def test_subject_is_read_from_the_jwt_payload():
    assert subject_of(token_with({"sub": "user-42"})) == "user-42"


def test_a_token_without_sub_is_rejected():
    with pytest.raises(IdentityError, match="sub"):
        subject_of(token_with({"email": "a@b"}))


def test_a_malformed_token_is_rejected():
    with pytest.raises(IdentityError):
        subject_of("not-a-jwt")


# --- behaviour simulation -------------------------------------------------------------


def test_topic_is_recovered_from_the_tag_string():
    assert topic_of(article("a1", "машинное обучение, нейросети")) == "machine-learning"
    assert topic_of(article("a2", "tag, t")) is None


def test_planned_events_respect_the_requested_count():
    by_topic = {t: [article(f"{t}-{i}", "") for i in range(5)] for t in TOPIC_NAMES[:4]}

    events = plan_events("u1", TOPIC_NAMES[:2], by_topic, 50, 14, random.Random(0), NOW)

    assert len(events) == 50


def test_preference_is_visible_in_the_generated_stream():
    # The whole point: a model must be able to recover which topics the user likes. If the
    # generator does not encode a preference, NDCG measures nothing.
    by_topic = {t: [article(f"{t}-{i}", "") for i in range(5)] for t in TOPIC_NAMES}
    preferred = TOPIC_NAMES[:2]

    events = plan_events("u1", preferred, by_topic, 400, 14, random.Random(1), NOW)

    preferred_ids = {a.id for t in preferred for a in by_topic[t]}
    share = sum(1 for e in events if e.article_id in preferred_ids) / len(events)

    assert share > 0.6


def test_likes_only_land_on_preferred_topics():
    by_topic = {t: [article(f"{t}-{i}", "") for i in range(3)] for t in TOPIC_NAMES}
    preferred = TOPIC_NAMES[:2]

    events = plan_events("u1", preferred, by_topic, 300, 14, random.Random(2), NOW)

    preferred_ids = {a.id for t in preferred for a in by_topic[t]}
    likes = [e for e in events if e.event_type == "ArticleLiked"]

    assert likes, "the funnel should produce some likes at this volume"
    assert all(e.article_id in preferred_ids for e in likes)


def test_a_read_always_follows_a_click_on_the_same_article():
    by_topic = {t: [article(f"{t}-0", "")] for t in TOPIC_NAMES[:2]}

    events = plan_events("u1", TOPIC_NAMES[:1], by_topic, 100, 7, random.Random(3), NOW)

    for index, event in enumerate(events):
        if event.event_type == "ArticleRead":
            previous = events[index - 1]
            assert previous.article_id == event.article_id
            assert previous.occurred_at <= event.occurred_at


def test_events_are_spread_over_the_requested_window():
    by_topic = {t: [article(f"{t}-0", "")] for t in TOPIC_NAMES[:3]}

    events = plan_events("u1", TOPIC_NAMES[:1], by_topic, 200, 21, random.Random(4), NOW)
    span_days = (max(e.occurred_at for e in events) - min(e.occurred_at for e in events)).days

    # Without a spread, every time-decay term in trending and user embeddings is a constant.
    assert span_days > 7


def test_no_event_is_stamped_in_the_future():
    # Every aggregate filters on `occurred_at < as_of`. A future timestamp is not merely odd,
    # it is invisible — it silently removes part of the most recent day from trending and from
    # every training set.
    by_topic = {t: [article(f"{t}-0", "")] for t in TOPIC_NAMES[:3]}

    events = plan_events("u1", TOPIC_NAMES[:1], by_topic, 300, 21, random.Random(6), NOW)

    assert max(e.occurred_at for e in events) < NOW


def test_every_event_carries_a_unique_id():
    by_topic = {t: [article(f"{t}-0", "")] for t in TOPIC_NAMES[:3]}

    events = plan_events("u1", TOPIC_NAMES[:1], by_topic, 200, 7, random.Random(5), NOW)

    assert len({e.event_id for e in events}) == len(events)
