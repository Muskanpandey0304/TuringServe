"""JSON-lines logging (ARCH 6.16): one JSON object per line with ts, level, event, ..."""

from __future__ import annotations

import json
import logging
import sys
import time

_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}
_logger = logging.getLogger("turingserve")


class _JsonHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        payload = getattr(record, "payload", None)
        line = {"ts": round(record.created, 3), "level": record.levelname.lower(),
                "event": record.getMessage()}
        if payload:
            line.update(payload)
        try:
            self.stream.write(json.dumps(line, default=str) + "\n")
            self.flush()
        except Exception:  # never let logging kill the engine
            pass


def setup_logging(level: str = "info") -> None:
    _logger.handlers.clear()
    _logger.addHandler(_JsonHandler(sys.stderr))
    _logger.setLevel(_LEVELS.get(level, logging.INFO))
    _logger.propagate = False


def log(event: str, level: str = "info", **fields) -> None:
    _logger.log(_LEVELS.get(level, logging.INFO), event, extra={"payload": fields})


def log_time() -> float:
    return time.time()
