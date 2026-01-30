"""
Model-specific API handling and configuration.
"""

import openai
from typing import Dict, Any


def create_client(model_id: str, api_key: str, use_openrouter: bool) -> tuple[openai.AsyncOpenAI, str]:
    """
    Create API client and return (client, final_model_id).

    Args:
        model_id: Model identifier
        api_key: API key
        use_openrouter: Whether to use OpenRouter API

    Returns:
        Tuple of (client, final_model_id)
    """
    if use_openrouter:
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/anthropics/llm-human-rights",
                "X-Title": "LLM Human Rights Research"
            }
        )
        final_model_id = model_id
    else:
        client = openai.AsyncOpenAI(api_key=api_key)
        final_model_id = model_id.split('/')[-1] if '/' in model_id else model_id

    return client, final_model_id


def build_messages(
    system_prompt: str,
    user_prompt: str,
    model_id: str
) -> list[Dict[str, str]]:
    """
    Build messages array for API call, handling model-specific quirks.

    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        model_id: Model identifier

    Returns:
        List of message dictionaries
    """
    # Gemini doesn't support system messages well - combine into user message
    if "gemini" in model_id.lower():
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        return [{"role": "user", "content": combined_prompt}]
    else:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


def build_multiturn_messages(
    system_prompt: str,
    conversation_turns: list[Dict[str, str]],
    model_id: str
) -> list[Dict[str, str]]:
    """
    Build multi-turn conversation messages array, handling model-specific quirks.

    Args:
        system_prompt: System prompt
        conversation_turns: List of conversation turns with {"role": "user"|"assistant", "content": "..."}
        model_id: Model identifier

    Returns:
        List of message dictionaries for multi-turn conversation

    Example:
        conversation_turns = [
            {"role": "user", "content": "Rate this case..."},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Are you sure?"}
        ]
    """
    # Gemini doesn't support system messages well - prepend to first user message
    if "gemini" in model_id.lower():
        messages = []
        system_prepended = False

        for turn in conversation_turns:
            if turn["role"] == "user" and not system_prepended:
                # Prepend system prompt to first user message
                combined_content = f"{system_prompt}\n\n{turn['content']}"
                messages.append({"role": "user", "content": combined_content})
                system_prepended = True
            else:
                messages.append(turn)

        return messages
    else:
        # Standard format: system message + conversation turns
        return [{"role": "system", "content": system_prompt}] + conversation_turns


def build_api_params(
    model_id: str,
    messages: list[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 100
) -> Dict[str, Any]:
    """
    Build API parameters, handling model-specific requirements.

    Args:
        model_id: Model identifier
        messages: Messages array
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Dictionary of API parameters
    """
    api_params = {
        "model": model_id,
        "messages": messages
    }

    # GPT-5 models use max_completion_tokens and only support temperature=1
    if "gpt-5" in model_id.lower():
        api_params["max_completion_tokens"] = 2048
        api_params["reasoning_effort"] = "low"
        # Don't set temperature for GPT-5 (only supports default value of 1)

    # Gemini uses internal reasoning tokens that count against max_tokens
    elif "gemini" in model_id.lower():
        # Use high max_tokens to ensure room for both reasoning and actual response
        # Note: reasoning control parameters may not work through OpenRouter
        api_params["max_tokens"] = 4096
        api_params["temperature"] = temperature

    # Other models
    else:
        api_params["max_tokens"] = max_tokens
        api_params["temperature"] = temperature

    return api_params


def get_response_content(response: Any) -> str | None:
    """
    Extract content from API response.

    Args:
        response: API response object

    Returns:
        Response content string or None if empty
    """
    if not response.choices or not response.choices[0].message.content:
        return None
    return response.choices[0].message.content.strip()
