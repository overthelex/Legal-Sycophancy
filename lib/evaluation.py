"""
Shared evaluation functions for ECHR case evaluation.
"""

import asyncio
import logging
import re
from typing import Tuple, List
import numpy as np

from .models import create_client, build_messages, build_api_params, get_response_content

logger = logging.getLogger(__name__)


def extract_rating_from_response(response: str) -> int:
    """
    Extract rating (1-5) from LLM response.

    Args:
        response: LLM response text

    Returns:
        Rating (1-5), defaults to 3 if cannot parse
    """
    # Try to find first digit 1-5
    match = re.search(r'[1-5]', response)
    if match:
        return int(match.group())

    # Default to neutral if can't parse
    logger.warning(f"Could not parse rating from response: {response[:100]}")
    return 3


async def evaluate_case(
    case_text: str,
    article: str,
    article_title: str,
    model_id: str,
    api_key: str,
    use_openrouter: bool,
    system_prompt: str,
    user_prompt_template: str,
    temperature: float = 1.0,
    max_retries: int = 3
) -> int:
    """
    Evaluate a single case and return rating (1-5).

    Args:
        case_text: Case text (full or summary)
        article: Article number (e.g., "3", "P1-1")
        article_title: Article title
        model_id: Model identifier
        api_key: API key
        use_openrouter: Whether to use OpenRouter
        system_prompt: System prompt
        user_prompt_template: User prompt template with {case_text}, {article}, {article_title} placeholders
        temperature: Sampling temperature
        max_retries: Maximum retry attempts

    Returns:
        Rating (1-5)
    """
    # Create client
    client, final_model_id = create_client(model_id, api_key, use_openrouter)

    # Format prompt
    user_prompt = user_prompt_template.format(
        case_text=case_text,
        article=article,
        article_title=article_title
    )

    # Build messages
    messages = build_messages(system_prompt, user_prompt, final_model_id)

    # Retry loop
    for attempt in range(max_retries):
        try:
            # Build API parameters
            api_params = build_api_params(final_model_id, messages, temperature)

            # Make API call
            response = await client.chat.completions.create(**api_params)

            # Extract content
            content = get_response_content(response)
            if not content:
                logger.warning(f"Empty response from API for model {final_model_id}")
                return 3

            # Extract rating
            rating = extract_rating_from_response(content)
            return rating

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Error evaluating case (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Error evaluating case after {max_retries} retries: {e}")
                return 3

    return 3


async def evaluate_case_with_sampling(
    case_text: str,
    article: str,
    article_title: str,
    model_id: str,
    api_key: str,
    use_openrouter: bool,
    system_prompt: str,
    user_prompt_template: str,
    num_samples: int = 5,
    sample_temperature: float = 1.0
) -> Tuple[float, List[int]]:
    """
    Evaluate a case multiple times with temperature sampling.

    Args:
        case_text: Case text (full or summary)
        article: Article number
        article_title: Article title
        model_id: Model identifier
        api_key: API key
        use_openrouter: Whether to use OpenRouter
        system_prompt: System prompt
        user_prompt_template: User prompt template
        num_samples: Number of samples
        sample_temperature: Sampling temperature

    Returns:
        Tuple of (average_rating, list_of_sample_ratings)
    """
    tasks = [
        evaluate_case(
            case_text, article, article_title, model_id, api_key,
            use_openrouter, system_prompt, user_prompt_template,
            sample_temperature
        )
        for _ in range(num_samples)
    ]

    sample_ratings = await asyncio.gather(*tasks)
    avg_rating = np.mean(sample_ratings)

    return avg_rating, sample_ratings


async def call_llm(
    client,
    model: str,
    prompt: str,
    max_retries: int = 3
) -> str:
    """
    Call LLM API with retry logic for summarization tasks.

    Args:
        client: OpenAI client
        model: Model identifier
        prompt: Prompt text
        max_retries: Maximum retry attempts

    Returns:
        LLM response text
    """
    for attempt in range(max_retries):
        try:
            # Build API parameters
            api_params = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }

            # GPT-5 models use max_completion_tokens instead of max_tokens
            if "gpt-5" in model.lower():
                api_params["max_completion_tokens"] = 2048
                api_params["reasoning_effort"] = "low"
                # Don't set temperature - GPT-5 only supports default value of 1
            else:
                api_params["max_tokens"] = 2000
                api_params["temperature"] = 0.3  # Low temperature for consistency

            response = await client.chat.completions.create(**api_params)

            if not response.choices or not response.choices[0].message.content:
                logger.warning(f"Empty response from API (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return ""

            return response.choices[0].message.content.strip()

        except Exception as e:
            if "RateLimitError" in str(type(e).__name__):
                wait_time = 5 * (2 ** attempt)
                wait_time = min(wait_time, 60)

                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} retries: {e}")
                    raise
            else:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Error calling LLM (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Error calling LLM after {max_retries} retries: {e}")
                    raise

    return ""
