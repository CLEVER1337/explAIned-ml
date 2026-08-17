"""Logging + exit codes shared by every job.

The codes mirror explAIned-faiss's index builder, because the same systemd units and the same
operator habits apply to both repositories:

    0  did the work
    1  nothing to do — an empty input is not a failure, and never destroys a good artifact
    2  infrastructure is down (Redis, ClickHouse, the article service)
"""

import logging

EXIT_OK = 0
EXIT_NOTHING_TO_DO = 1
EXIT_INFRA = 2

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT)
