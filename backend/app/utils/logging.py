"""
Structured logging and performance latency timing utility.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configure root bms logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("bms")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming request method, path, execution latency in milliseconds, and status code."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Attach custom process time header
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

        if not request.url.path.endswith("/health"):
            logger.info(
                f"{request.method} {request.url.path} -> {response.status_code} "
                f"({process_time_ms:.2f}ms)"
            )
        return response
