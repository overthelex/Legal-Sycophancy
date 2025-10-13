from datetime import datetime
import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from src.client import LLMClient
from src.translate import TranslationClient
from src.utils import _log_token_usage
from inspect_ai.model import GenerateConfig

import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


async def run_evaluation_pipeline(
    model_id: str,
    generation_config: GenerateConfig,
    judge_model_id: str,
    judge_generation_config: GenerateConfig,
    eval_method: str,
    eval_persona: str,
    language: str,
    scenarios_df: pd.DataFrame,
    default_language: str,
    system_prompt: str,
    prompt_template: str,
    open_judge_system_prompt: str,
    open_judge_prompt_template: str,
    classification_to_score_map: dict,
    num_samples: int = 5,
    log_usage: bool = True,
) -> pd.DataFrame:
    """
    Run the evaluation pipeline for the given scenarios.

    Args:
        model_id: Model identifier (e.g., 'openai/gpt-4.1-2025-04-14', 'openrouter/qwen/qwen3-32b')
    """

    client = LLMClient(model_id, config=generation_config)
    judge_client = LLMClient(judge_model_id, config=judge_generation_config)
    translator = TranslationClient()

    out_df = scenarios_df.copy()

    out_df["lang_code"] = language

    if eval_method == "likert":
        pass  # TODO
    elif eval_method == "open_ended":
        out_df["system_prompt"] = system_prompt
        out_df["scenario_prompt"] = out_df.apply(
            lambda row: prompt_template.format(
                limited_article=row["limited_article"],
                limited_article_name=row["limited_article_name"],
                scenario_text=row["scenario_text"],
            ),
            axis=1,
        )

    # Translate prompts to target language
    if language == default_language:
        out_df["scenario_text_translated"] = out_df["scenario_text"]
        # Assume same system prompt for all
        # TODO: should make system_prompt_translated column
        # out_df["system_prompt_translated"] = out_df["system_prompt"]
    else:
        out_df["scenario_text_translated"] = await translator.translate_batch(
            out_df["scenario_text"].tolist(),
            source_lang=default_language,
            target_lang=language,
            desc=f"Translating scenario texts to {language}",
        )
        # TODO: should make system_prompt_translated column
        out_df["system_prompt"] = await translator.translate(
            out_df["system_prompt"].iloc[0],  # Assume same system prompt for all
            source_lang=default_language,
            target_lang=language,
        )
        # Can support unique system prompts if needed
        # out_df["system_prompt"] = await translator.translate_batch(
        #     out_df["system_prompt"].tolist(), source_lang=default_language, target_lang=language
        # )
        out_df["scenario_prompt"] = await translator.translate_batch(
            out_df["scenario_prompt"].tolist(),
            source_lang=default_language,
            target_lang=language,
            desc=f"Translating scenario prompts to {language}",
        )

    # Repeat each row num_samples times with sample_id
    repeated = out_df.loc[np.repeat(out_df.index, num_samples)].reset_index(drop=True)
    repeated["sample_id"] = np.tile(np.arange(num_samples), len(out_df))
    out_df = repeated

    # Generate responses for scenarios
    raw_responses = await client.generate_batch(
        prompts=out_df["scenario_prompt"].tolist(),
        system_messages=out_df["system_prompt"].fillna("").tolist(),
        desc=f"Generating {model_id} responses in {language}",
    )

    # Translate responses back to English if needed
    if language == default_language:
        translated_responses = raw_responses
    else:
        translated_responses = await translator.translate_batch(
            texts=raw_responses,
            target_lang=default_language,
            source_lang=language,
            desc=f"Translating responses to {default_language}",
        )

    # Evaluate responses with judge model
    scenarios = out_df["scenario_text"].tolist()
    judge_prompts = [
        open_judge_prompt_template.format(
            scenario_text=scenario, response_for_judge=response
        )
        for scenario, response in zip(scenarios, translated_responses)
    ]
    judge_system_prompts = [open_judge_system_prompt for _ in judge_prompts]
    judge_raw_responses = await judge_client.generate_batch(
        prompts=judge_prompts,
        system_messages=judge_system_prompts,
        desc=f"Evaluating responses with {judge_model_id}",
    )

    # Parse judge responses
    classifications = []
    scores = []
    reasonings = []
    for eval_response in judge_raw_responses:
        # Clean eval response
        eval_response = eval_response.replace("```json", "").replace("```", "").strip()

        try:
            eval_json = json.loads(eval_response)
            classifications.append(eval_json.get("classification", "unknown"))
            scores.append(
                classification_to_score_map.get(
                    eval_json.get("classification", "unknown"), 0
                )
            )
            reasonings.append(eval_json.get("reasoning", ""))
        except json.JSONDecodeError:
            classifications.append("unknown")
            scores.append(0)
            reasonings.append("")

    # Compile results into DataFrame
    out_df["model_id"] = model_id
    out_df["eval_method"] = eval_method
    out_df["eval_persona"] = eval_persona
    out_df["raw_response"] = raw_responses
    out_df["translated_response"] = translated_responses
    out_df["judge_model_id"] = judge_model_id
    out_df["judge_raw_response"] = judge_raw_responses
    out_df["judge_classification"] = classifications
    out_df["judge_score"] = scores
    out_df["judge_reasoning"] = reasonings

    usage_log_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_eval_usage.log"
    _log_token_usage(client.usage, usage_log_name, model_id)
    _log_token_usage(judge_client.usage, usage_log_name, judge_model_id)

    # Simple stats
    avg_score = out_df["judge_score"].mean()
    std_score = out_df["judge_score"].std()
    logger.info(
        f"📍 Average judge score for model {model_id}: {avg_score:.2f} ± {std_score:.2f}"
    )

    # distribution across scores
    score_counts = out_df["judge_score"].value_counts().sort_index()
    logger.info(f"📍 Score distribution for model {model_id}:\n{score_counts}")

    return out_df
