"""
Shared evaluation functions for ECHR case evaluation.
"""

import asyncio
import logging
import re
from typing import Tuple, List, Dict, Any
import numpy as np

from .models import create_client, build_messages, build_multiturn_messages, build_api_params, get_response_content, prepare_request

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


# ============================================================================
# Queue-based evaluation functions for efficient parallel execution
# ============================================================================

def prepare_evaluation_requests(
    cases: List[Dict[str, Any]],
    model_id: str,
    api_key: str,
    use_openrouter: bool,
    system_prompt: str,
    user_prompt_template: str,
    num_samples: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 100,
    case_text_key: str = 'full_case_text'
) -> List[Dict[str, Any]]:
    """
    Pre-generate all evaluation requests for queue-based execution.

    Args:
        cases: List of case dictionaries with case data
        model_id: Model identifier
        api_key: API key
        use_openrouter: Whether to use OpenRouter
        system_prompt: System prompt
        user_prompt_template: User prompt template with {case_text}, {article}, {article_title}
        num_samples: Number of samples per case (default: 1)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        case_text_key: Key for case text in case dict (default: 'full_case_text', can be 'summary')

    Returns:
        List of request dictionaries ready for QueuedLLMClient
    """
    requests = []

    # Article titles mapping
    article_titles = {
        "2": "Right to life",
        "3": "Prohibition of torture",
        "5": "Right to liberty and security",
        "6": "Right to a fair trial",
        "8": "Right to respect for private and family life",
        "10": "Freedom of expression",
        "14": "Prohibition of discrimination",
        "P1-1": "Protection of property"
    }

    for case in cases:
        article = case['article']
        article_title = article_titles.get(article, f"Article {article}")
        case_text = case.get(case_text_key, case.get('full_case_text', ''))

        # Generate multiple samples if requested
        for sample_idx in range(num_samples):
            # Format user prompt
            user_prompt = user_prompt_template.format(
                case_text=case_text,
                article=article,
                article_title=article_title
            )

            # Build messages
            messages = build_messages(system_prompt, user_prompt, model_id)

            # Prepare request
            custom_id = f"{case.get('item_id', case.get('case_name', 'unknown'))}_sample_{sample_idx}"

            request = prepare_request(
                custom_id=custom_id,
                model_id=model_id,
                messages=messages,
                api_key=api_key,
                use_openrouter=use_openrouter,
                temperature=temperature,
                max_tokens=max_tokens
            )

            requests.append(request)

    logger.info(f"Prepared {len(requests)} evaluation requests ({len(cases)} cases × {num_samples} samples)")
    return requests


def prepare_multiturn_requests(
    initial_results: Dict[str, str],
    followup_prompt: str,
    model_id: str,
    api_key: str,
    use_openrouter: bool,
    system_prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 100
) -> List[Dict[str, Any]]:
    """
    Pre-generate multi-turn conversation requests (for RQ3 confidence challenge).

    Args:
        initial_results: Dict mapping custom_id to initial response
        followup_prompt: Follow-up prompt (e.g., "Are you sure?")
        model_id: Model identifier
        api_key: API key
        use_openrouter: Whether to use OpenRouter
        system_prompt: System prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens

    Returns:
        List of request dictionaries ready for QueuedLLMClient
    """
    requests = []

    for custom_id, initial_response in initial_results.items():
        # Build multi-turn conversation
        conversation_turns = [
            {"role": "assistant", "content": initial_response},
            {"role": "user", "content": followup_prompt}
        ]

        # Build messages (includes system prompt + conversation)
        messages = build_multiturn_messages(system_prompt, conversation_turns, model_id)

        # Prepare request
        followup_id = f"{custom_id}_followup"

        request = prepare_request(
            custom_id=followup_id,
            model_id=model_id,
            messages=messages,
            api_key=api_key,
            use_openrouter=use_openrouter,
            temperature=temperature,
            max_tokens=max_tokens
        )

        requests.append(request)

    logger.info(f"Prepared {len(requests)} multi-turn follow-up requests")
    return requests


def process_evaluation_results(
    results: Dict[str, str],
    cases: List[Dict[str, Any]],
    num_samples: int = 1
) -> List[Dict[str, Any]]:
    """
    Process queue results and extract ratings for each case.

    Args:
        results: Dict mapping custom_id to response content from QueuedLLMClient
        cases: List of original case dictionaries
        num_samples: Number of samples per case

    Returns:
        List of case results with ratings, format:
        [
            {
                'item_id': '...',
                'case_name': '...',
                'article': '...',
                'violation_label': '...',
                'sample_ratings': [1, 2, 1, ...],
                'avg_rating': 1.5,
                'num_abstentions': 0
            },
            ...
        ]
    """
    case_results = []

    for case in cases:
        item_id = case.get('item_id', case.get('case_name', 'unknown'))

        # Collect all sample ratings for this case
        sample_ratings = []
        for sample_idx in range(num_samples):
            custom_id = f"{item_id}_sample_{sample_idx}"

            if custom_id in results:
                response = results[custom_id]
                rating = extract_rating_from_response(response)
                sample_ratings.append(rating)
            else:
                logger.warning(f"Missing result for {custom_id}, using default rating 3")
                sample_ratings.append(3)

        # Calculate aggregated metrics
        avg_rating = np.mean(sample_ratings) if sample_ratings else 3.0
        num_abstentions = sum(1 for r in sample_ratings if r == 3)

        case_result = {
            'item_id': case.get('item_id', ''),
            'case_name': case.get('case_name', ''),
            'article': case['article'],
            'violation_label': case['violation_label'],
            'sample_ratings': sample_ratings,
            'avg_rating': float(avg_rating),
            'num_abstentions': num_abstentions
        }

        case_results.append(case_result)

    logger.info(f"Processed results for {len(case_results)} cases")
    return case_results
