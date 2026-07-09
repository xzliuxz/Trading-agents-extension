from __future__ import annotations
import logging
import time
import random
from typing import Any, Optional
logger = logging.getLogger(__name__)

def _fallback_message(content: str):
    try:
        from langchain_core.messages import AIMessage
        return AIMessage(content=content, id='fallback')
    except Exception:

        class _SimpleMessage:

            def __init__(self, content: str):
                self.content = content
                self.id = 'fallback'
        return _SimpleMessage(content)
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
_RETRYABLE_PATTERNS = ['rate_limit', 'rate limit', '429', 'too many requests', 'resource_exhausted', 'overloaded', 'server_error', 'internal server error', '500', '502', '503', '504', 'timeout', 'timed out', 'connection', 'temporary', 'unavailable', 'deadline exceeded', 'quota']

def _is_retryable(error: Exception) -> bool:
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    for pattern in _RETRYABLE_PATTERNS:
        if pattern in error_str or pattern in error_type:
            return True
    status = getattr(error, 'status_code', None) or getattr(error, 'code', None)
    if status and isinstance(status, int):
        if status == 429 or status >= 500:
            return True
    return False

def invoke_with_retry(llm, input_data: Any, *, max_retries: int=DEFAULT_MAX_RETRIES, base_delay: float=DEFAULT_BASE_DELAY, max_delay: float=DEFAULT_MAX_DELAY, fallback_content: str='', caller_name: str='') -> Any:
    label = f"[LLM Retry{(f' / {caller_name}' if caller_name else '')}]"
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            result = llm.invoke(input_data)
            if attempt > 0:
                logger.info('%s Succeeded on attempt %d/%d', label, attempt + 1, max_retries + 1)
            return result
        except Exception as e:
            last_error = e
            retryable = _is_retryable(e)
            if attempt >= max_retries or not retryable:
                logger.error('%s Failed after %d attempt(s): %s [retryable=%s]', label, attempt + 1, e, retryable)
                break
            delay = min(base_delay * 2 ** attempt + random.uniform(0, 1), max_delay)
            logger.warning('%s Attempt %d/%d failed (%s: %s). Retrying in %.1fs...', label, attempt + 1, max_retries + 1, type(e).__name__, str(e)[:120], delay)
            time.sleep(delay)
    if fallback_content:
        logger.warning('%s All retries exhausted. Returning fallback content.', label)
        return _fallback_message(fallback_content)
    raise last_error
