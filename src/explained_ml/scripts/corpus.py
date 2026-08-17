"""A synthetic but *topically separable* article corpus.

Why this exists: the live corpus is 796 rows whose bodies are literally the word "body", with
one author and three distinct tag strings. Sentence-BERT over that produces noise, ALS over one
user produces nothing, and NDCG on either is unmeasurable. Nothing downstream of embeddings can
be judged until the corpus has structure.

The structure is deliberate and small: ten topics with disjoint vocabularies. That makes
embeddings visibly cluster (so `/search` results are reviewable by eye), gives ALS a signal it
is *supposed* to recover, and lets `simulate_behavior` express a preference that a model either
finds or does not. It is a fixture, not data — see the README.
"""

import random
from dataclasses import dataclass

TOPICS: dict[str, dict[str, list[str]]] = {
    "distributed-systems": {
        "tags": ["распределённые системы", "консенсус", "отказоустойчивость"],
        "nouns": ["кворум", "реплика", "партиция", "лидер", "журнал", "снапшот", "координатор"],
        "verbs": ["реплицирует", "фиксирует", "переизбирает", "восстанавливает", "согласует"],
        "topics": ["Raft", "двухфазный коммит", "split-brain", "векторные часы", "gossip-протокол"],
    },
    "machine-learning": {
        "tags": ["машинное обучение", "нейросети", "ранжирование"],
        "nouns": ["признак", "эмбеддинг", "градиент", "выборка", "метрика", "регуляризация"],
        "verbs": ["обучает", "переобучается", "сходится", "предсказывает", "калибрует"],
        "topics": ["градиентный бустинг", "трансформеры", "функция потерь", "NDCG", "переобучение"],
    },
    "frontend": {
        "tags": ["фронтенд", "браузер", "интерфейсы"],
        "nouns": ["компонент", "состояние", "рендер", "стиль", "событие", "виртуальный DOM"],
        "verbs": ["перерисовывает", "монтирует", "подписывается", "кэширует", "гидрирует"],
        "topics": ["реактивность", "серверный рендеринг", "доступность", "анимации", "бандлы"],
    },
    "databases": {
        "tags": ["базы данных", "sql", "индексы"],
        "nouns": ["индекс", "транзакция", "план запроса", "блокировка", "секция", "буфер"],
        "verbs": ["сканирует", "блокирует", "вакуумирует", "материализует", "сортирует"],
        "topics": ["B-дерево", "уровни изоляции", "MVCC", "шардирование", "оконные функции"],
    },
    "security": {
        "tags": ["безопасность", "криптография", "аутентификация"],
        "nouns": ["токен", "подпись", "ключ", "сертификат", "уязвимость", "сессия"],
        "verbs": ["подписывает", "валидирует", "ротирует", "шифрует", "отзывает"],
        "topics": ["JWT", "OAuth", "инъекции", "цепочка доверия", "утечки секретов"],
    },
    "devops": {
        "tags": ["devops", "инфраструктура", "мониторинг"],
        "nouns": ["контейнер", "пайплайн", "метрика", "алерт", "дашборд", "деплой"],
        "verbs": ["раскатывает", "откатывает", "масштабирует", "собирает", "оповещает"],
        "topics": ["голубо-зелёный деплой", "SLO", "трассировка", "автоскейлинг", "хаос-инженерия"],
    },
    "mobile": {
        "tags": ["мобильная разработка", "android", "ios"],
        "nouns": ["экран", "жест", "уведомление", "батарея", "кэш", "разрешение"],
        "verbs": ["анимирует", "синхронизирует", "экономит", "запрашивает", "восстанавливает"],
        "topics": ["офлайн-режим", "фоновые задачи", "адаптивная вёрстка", "push-уведомления"],
    },
    "gamedev": {
        "tags": ["геймдев", "графика", "движки"],
        "nouns": ["спрайт", "шейдер", "коллизия", "кадр", "сцена", "физика"],
        "verbs": ["рендерит", "симулирует", "интерполирует", "оптимизирует", "запекает"],
        "topics": ["игровой цикл", "процедурная генерация", "освещение", "неткод", "тайминги"],
    },
    "product": {
        "tags": ["продукт", "аналитика", "исследования"],
        "nouns": ["гипотеза", "метрика", "воронка", "когорта", "интервью", "retention"],
        "verbs": ["проверяет", "измеряет", "сегментирует", "приоритизирует", "валидирует"],
        "topics": ["A/B-тесты", "unit-экономика", "user story", "discovery", "продуктовый долг"],
    },
    "hardware": {
        "tags": ["железо", "производительность", "низкий уровень"],
        "nouns": ["кэш-линия", "регистр", "конвейер", "прерывание", "шина", "ядро"],
        "verbs": ["префетчит", "выравнивает", "вытесняет", "переупорядочивает", "измеряет"],
        "topics": ["ложное разделение", "NUMA", "векторизация", "промахи кэша", "барьеры памяти"],
    },
}

TOPIC_NAMES = list(TOPICS)

_TITLE_SHAPES = [
    "{Topic}: что ломается на практике",
    "Как мы чинили {topic}",
    "{Topic} без магии",
    "Разбор: {topic} под нагрузкой",
    "Почему {topic} обманывает интуицию",
    "{Topic} — три ошибки и один вывод",
]

_SENTENCES = [
    "{Noun} {verb} на каждом шаге, и это заметно в профиле.",
    "Когда {noun} перестаёт справляться, {topic} становится узким местом.",
    "Мы измерили, как {noun} ведёт себя под нагрузкой, и результат оказался неочевидным.",
    "Практика показывает: {noun} {verb} ровно до того момента, пока нагрузка не удвоится.",
    "В теории {topic} выглядит просто, но {noun} добавляет неожиданные ограничения.",
    "Отладка началась с того, что {noun} {verb} не так, как описано в документации.",
    "Мы сравнили два подхода к {topic} и оставили тот, который проще откатить.",
    "Главный вывод: {noun} нужно наблюдать, а не угадывать.",
]


@dataclass(frozen=True, slots=True)
class DraftArticle:
    title: str
    description: str
    content: str
    tags: str
    topic: str


def generate(count: int, seed: int = 42) -> list[DraftArticle]:
    """Deterministic for a given (count, seed) — a rerun produces the same corpus."""
    rng = random.Random(seed)
    articles: list[DraftArticle] = []

    for index in range(count):
        topic = TOPIC_NAMES[index % len(TOPIC_NAMES)]
        articles.append(_one(topic, rng))

    return articles


def _one(topic: str, rng: random.Random) -> DraftArticle:
    spec = TOPICS[topic]
    subject = rng.choice(spec["topics"])

    title = rng.choice(_TITLE_SHAPES).format(topic=subject, Topic=subject.capitalize())
    paragraphs = [_paragraph(spec, rng) for _ in range(rng.randint(5, 9))]
    description = _sentence(spec, rng)
    tags = ", ".join(rng.sample(spec["tags"], k=min(len(spec["tags"]), rng.randint(2, 3))))

    return DraftArticle(
        title=title,
        description=description,
        content="\n\n".join(paragraphs),
        tags=tags,
        topic=topic,
    )


def _paragraph(spec: dict[str, list[str]], rng: random.Random) -> str:
    return " ".join(_sentence(spec, rng) for _ in range(rng.randint(3, 6)))


def _sentence(spec: dict[str, list[str]], rng: random.Random) -> str:
    noun = rng.choice(spec["nouns"])
    return rng.choice(_SENTENCES).format(
        noun=noun,
        Noun=noun.capitalize(),
        verb=rng.choice(spec["verbs"]),
        topic=rng.choice(spec["topics"]),
    )
