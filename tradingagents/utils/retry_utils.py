import logging
from typing import Any
from google.api_core.exceptions import InternalServerError, ServiceUnavailable
try:
    from google.genai.errors import ServerError as GenAIServerError
except ImportError:
    GenAIServerError = None
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    before_sleep_log
)

logger = logging.getLogger(__name__)


class EmptyResponseError(Exception):
    """Raised when LLM returns empty response with no tool calls and no content."""
    pass


def should_retry(exception):
    """Determine if an exception should trigger a retry."""
    retryable_exceptions = [InternalServerError, ServiceUnavailable, EmptyResponseError]

    # Add GenAI ServerError if available
    if GenAIServerError is not None:
        retryable_exceptions.append(GenAIServerError)

    # Check if it's a retryable exception
    if isinstance(exception, tuple(retryable_exceptions)):
        logger.warning(f"Retrying on {type(exception).__name__}: {str(exception)}")
        return True

    # Also retry on any Google API error with code 500-599
    if hasattr(exception, 'code') and exception.code and 500 <= exception.code < 600:
        logger.warning(f"Retrying on server error with code {exception.code}: {str(exception)}")
        return True

    logger.debug(f"Not retrying exception {type(exception).__name__}: {str(exception)}")
    return False


def create_retry_wrapper(
    max_attempts: int = 5,
    initial_wait: float = 2.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0
):
    """Create a retry decorator for Google AI API calls with exponential backoff."""

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=initial_wait,
            max=max_wait
        ),
        retry=retry_if_exception(should_retry),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )


class RetryableLLM:
    """Wrapper for LLM with automatic retry logic for transient errors."""

    def __init__(
        self,
        llm: Any,
        max_attempts: int = 7,
        initial_wait: float = 5.0,
        max_wait: float = 120.0,
        multiplier: float = 2.0
    ):
        self.llm = llm
        self.retry_decorator = create_retry_wrapper(
            max_attempts=max_attempts,
            initial_wait=initial_wait,
            max_wait=max_wait,
            multiplier=multiplier
        )

    def invoke(self, *args, **kwargs):
        """Invoke the LLM with automatic retry on transient errors and empty responses."""

        @self.retry_decorator
        def _invoke_with_retry():
            result = self.llm.invoke(*args, **kwargs)

            # Check for empty response
            if hasattr(result, 'tool_calls') and hasattr(result, 'content'):
                if len(result.tool_calls) == 0 and isinstance(result.content, str) and len(result.content.strip()) == 0:
                    logger.warning("LLM returned empty response (no tool calls and empty string content). Retrying...")
                    raise EmptyResponseError("LLM returned empty response")

            # Check for MAX_TOKENS finish reason
            if hasattr(result, 'response_metadata') and isinstance(result.response_metadata, dict):
                finish_reason = result.response_metadata.get('finish_reason', '')
                if finish_reason == 'MAX_TOKENS':
                    logger.warning(f"LLM hit token limit (finish_reason: MAX_TOKENS). Content length: {len(result.content) if hasattr(result, 'content') else 'N/A'}. Retrying...")
                    raise EmptyResponseError(f"LLM hit token limit - finish_reason: {finish_reason}")

            return result

        return _invoke_with_retry()

    def __getattr__(self, name):
        """Proxy other attributes to the wrapped LLM."""
        return getattr(self.llm, name)
