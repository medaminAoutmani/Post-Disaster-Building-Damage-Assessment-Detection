import logging
import sys

try:
    import structlog

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
except Exception:  # pragma: no cover - only used before dependencies are installed
    structlog = None
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class StdlibLoggerAdapter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def _log(self, level: int, event: str, **kwargs):
        extra = f" {kwargs}" if kwargs else ""
        self.logger.log(level, "%s%s", event, extra)

    def info(self, event: str, **kwargs):
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs):
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs):
        self._log(logging.ERROR, event, **kwargs)

    def debug(self, event: str, **kwargs):
        self._log(logging.DEBUG, event, **kwargs)

def get_logger(name: str):
    if structlog is not None:
        return structlog.get_logger(name)
    return StdlibLoggerAdapter(logging.getLogger(name))
