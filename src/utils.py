import os
from typing import List
from inspect_ai.model import ModelUsage
from tqdm.asyncio import tqdm


async def tqdm_gather(*fs, desc: str, return_exceptions=False, **kwargs):
    if not return_exceptions:
        return await tqdm.gather(*fs, desc=desc, **kwargs)

    async def wrap(f):
        try:
            return await f
        except Exception as e:
            return e

    return await tqdm.gather(*map(wrap, fs), desc=desc, **kwargs)


def slashes_to_dashes(s: str) -> str:
    return s.replace("/", "-").replace("\\", "-")


def get_openrouter_extra_body(
    providers: List[str], allow_fallbacks: bool = False
) -> dict:
    # Messy fix to handle list of providers passed as a single string
    # if len(providers) == 1:
    #     providers = providers[0].split(" ")

    return {
        "provider": {
            "order": providers,
            "allow_fallbacks": allow_fallbacks,
        }
    }


def _log_token_usage(usage: ModelUsage, filename: str, model: str) -> None:
    to_save = f"""
==================================================
Model: "{model}"
  Total Prompt Tokens: {usage.input_tokens}
  Total Completion Tokens: {usage.output_tokens}
  Total Tokens: {usage.total_tokens}
  Cache Write: {usage.input_tokens_cache_write or 0}
  Cache Read: {usage.input_tokens_cache_read or 0}
  Reasoning Tokens: {usage.reasoning_tokens or 0}
==================================================
""".strip()

    path = f"logs/token_usage/{filename}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(to_save + "\n\n")


def get_model_config(model_id: str, config_file: str = "src/models.yaml") -> dict:
    import yaml

    with open(config_file, "r") as f:
        models = yaml.safe_load(f)

    if model_id not in models:
        raise ValueError(f"Model ID '{model_id}' not found in {config_file}.")

    return models[model_id]
