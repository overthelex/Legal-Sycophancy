"""
Unified queue-based LLM client for OpenAI and OpenRouter APIs.

Uses producer-consumer pattern with connection pooling for efficient
parallel execution of thousands of requests.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Tuple
import asyncio
import aiohttp
import json
import time
import logging

logger = logging.getLogger(__name__)


class QueuedLLMClient:
    """
    Execute LLM requests using continuous pipeline with Producer-Consumer pattern.

    Supports both OpenAI and OpenRouter APIs via aiohttp with connection pooling.
    Workers continuously pull from queue and process - no wave batching.
    Much more efficient than batch execution when requests have variable latency.
    """

    # API endpoints
    OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        max_workers: int = 50,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize queued LLM client.

        Args:
            max_workers: Number of concurrent worker tasks (default: 50)
            timeout: Request timeout in seconds (default: 120)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries

    def execute_all(
        self,
        requests_list: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, str]:
        """
        Execute all requests using continuous pipeline.

        This is a synchronous wrapper around the async implementation.

        Args:
            requests_list: List of request dictionaries with format:
                {
                    'custom_id': str,           # Unique identifier
                    'model': str,               # Model ID
                    'messages': List[Dict],     # Messages array
                    'api_key': str,            # API key to use
                    'backend': 'openai' | 'openrouter',  # Which endpoint
                    'temperature': float,       # Optional
                    'max_tokens': int,         # Optional
                    'max_completion_tokens': int,  # Optional (GPT-5)
                    'reasoning_effort': str,   # Optional (GPT-5, o1, o3)
                }
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            Dictionary mapping custom_id to response content string
        """
        return asyncio.run(self._execute_all_async(requests_list, progress_callback))

    async def _execute_all_async(
        self,
        requests_list: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, str]:
        """
        Async implementation of continuous pipeline execution.

        Producer: Puts all requests into queue upfront
        Consumers: N workers continuously pull and process

        Args:
            requests_list: List of request dictionaries
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary mapping custom_id to response content string
        """
        total_requests = len(requests_list)
        logger.info(f"Starting execution of {total_requests} requests with {self.max_workers} workers")

        # Create queue and results dict
        queue = asyncio.Queue()
        results = {}
        completed_count = 0
        results_lock = asyncio.Lock()

        # Configure connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_workers * 2,
            limit_per_host=self.max_workers,
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )

        timeout_obj = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout_obj, connector=connector) as session:
            # Producer: Add all requests to queue upfront
            for request in requests_list:
                await queue.put(request)

            # Add sentinel values to signal workers to stop
            for _ in range(self.max_workers):
                await queue.put(None)

            # Consumer: Create worker tasks
            async def worker():
                nonlocal completed_count

                while True:
                    request = await queue.get()

                    # Check for sentinel value
                    if request is None:
                        queue.task_done()
                        break

                    try:
                        # Execute request
                        custom_id, response_content = await self._execute_single_async(
                            session, request
                        )

                        # Store result
                        async with results_lock:
                            results[custom_id] = response_content
                            completed_count += 1

                            # Call progress callback
                            if progress_callback:
                                progress_callback(completed_count, total_requests)

                    except Exception as e:
                        # Log error but continue processing
                        custom_id = request.get('custom_id', 'unknown')
                        logger.error(f"Worker error for {custom_id}: {e}")

                        async with results_lock:
                            results[custom_id] = f"ERROR: {str(e)}"
                            completed_count += 1

                            if progress_callback:
                                progress_callback(completed_count, total_requests)

                    finally:
                        queue.task_done()

            # Start all workers
            workers = [asyncio.create_task(worker()) for _ in range(self.max_workers)]

            # Wait for all work to complete
            await queue.join()

            # Wait for all workers to finish
            await asyncio.gather(*workers)

        logger.info(f"Completed {completed_count}/{total_requests} requests")
        return results

    async def _execute_single_async(
        self,
        session: aiohttp.ClientSession,
        request: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Execute a single request asynchronously with retry logic.

        Args:
            session: aiohttp session
            request: Request dictionary

        Returns:
            Tuple of (custom_id, response_content)
        """
        custom_id = request['custom_id']
        backend = request['backend']

        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Determine endpoint and build headers
                if backend == 'openai':
                    endpoint = self.OPENAI_ENDPOINT
                    headers = {
                        "Authorization": f"Bearer {request['api_key']}",
                        "Content-Type": "application/json"
                    }
                elif backend == 'openrouter':
                    endpoint = self.OPENROUTER_ENDPOINT
                    headers = {
                        "Authorization": f"Bearer {request['api_key']}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/anthropics/llm-human-rights",
                        "X-Title": "LLM Human Rights Research"
                    }
                else:
                    raise ValueError(f"Unknown backend: {backend}")

                # Build request body
                data = {
                    "model": request['model'],
                    "messages": request['messages']
                }

                # Add optional parameters
                if 'temperature' in request:
                    data['temperature'] = request['temperature']

                if 'max_tokens' in request:
                    data['max_tokens'] = request['max_tokens']
                elif 'max_completion_tokens' in request:
                    data['max_completion_tokens'] = request['max_completion_tokens']

                if 'reasoning_effort' in request:
                    data['reasoning_effort'] = request['reasoning_effort']

                # Make async HTTP request
                async with session.post(endpoint, headers=headers, json=data) as response:
                    response.raise_for_status()
                    response_data = await response.json()

                    # Extract content
                    if not response_data.get("choices") or not response_data["choices"][0].get("message"):
                        logger.warning(f"Empty response for {custom_id}")
                        return custom_id, ""

                    content = response_data["choices"][0]["message"].get("content", "")
                    return custom_id, content

            except asyncio.TimeoutError as e:
                # Timeout - retryable
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    logger.warning(f"Timeout for {custom_id} (attempt {attempt + 1}/{self.max_retries}), waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded for {custom_id}: {str(e)}")
                    return custom_id, ""

            except aiohttp.ClientResponseError as e:
                # HTTP errors (rate limiting, server errors)
                if e.status in [429, 500, 502, 503, 504] and attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    logger.warning(f"HTTP {e.status} for {custom_id} (attempt {attempt + 1}/{self.max_retries}), waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"HTTP error for {custom_id}: {str(e)}")
                    return custom_id, ""

            except aiohttp.ClientError as e:
                # Other client errors - retryable
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    logger.warning(f"Client error for {custom_id} (attempt {attempt + 1}/{self.max_retries}), waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded for {custom_id}: {str(e)}")
                    return custom_id, ""

            except Exception as e:
                # Non-retryable errors
                logger.error(f"Unexpected error for {custom_id}: {str(e)}")
                return custom_id, ""

        # Should never reach here, but just in case
        return custom_id, ""
