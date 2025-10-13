""" """

import os
import argparse
import asyncio
import logging
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
from inspect_ai.model import GenerateConfig

from src.utils import slashes_to_dashes, get_openrouter_extra_body
from src.synthetic.config import (
    MAX_CONNECTIONS,
    MAX_RETRIES,
    DEFAULT_LANGUAGE,
    CLASSIFICATION_TO_SCORE_MAP,
)
from src.evaluation_pipeline import run_evaluation_pipeline
from src.synthetic.evaluation_prompts import (
    LIKERT_SYSTEM_PROMPT,
    LIKERT_PROMPT_TEMPLATE,
    OPEN_SYSTEM_PROMPT,
    OPEN_PROMPT_TEMPLATE,
    OPEN_JUDGE_SYSTEM_PROMPT,
    OPEN_JUDGE_PROMPT_TEMPLATE,
)


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

load_dotenv()
DEFAULT_MODEL = os.getenv("AZUREAI_OPENAI_GPT4.1", "openai/gpt-4.1-2025-04-14")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic human rights scenarios based on the UDHR or ECHR."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Target language code for translation (default: 'en').",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., 'openai/gpt-4.1-2025-04-14', 'openrouter/qwen/qwen3-32b').",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="*",
        help="List of providers for OpenRouter (e.g., 'deepinfra/fp8').",
    )
    parser.add_argument(
        "--eval_method",
        type=str,
        default="open_ended",
        choices=["likert", "open_ended"],
        help="Evaluation method: 'likert' or 'open_ended' (default: 'open_ended').",
    )
    parser.add_argument(
        "--eval_persona",
        type=str,
        default="none",
        choices=["none", "individual-rights", "government-power"],
        help="Steering instruction for evaluation (default: 'none').",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=DEFAULT_MODEL,
        help="Judge model identifier (e.g., 'openai/gpt-4.1-2025-04-14').",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of scenarios to generate per configuration (default: 5).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing scenario files."
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default="data/experiments/synthetic",
        help="Directory to retrieve generated scenarios and save model responses (default: 'data/experiments/synthetic').",
    )
    parser.add_argument(
        "--rights",
        type=str,
        default="udhr",
        choices=["udhr", "echr"],
        help="Rights to base scenarios on (default: 'udhr').",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with limited scenarios."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model = args.model
    providers = args.providers
    eval_method = args.eval_method
    eval_persona = args.eval_persona
    judge_model = args.judge_model
    language = args.language
    num_samples = args.num_samples
    working_dir = args.working_dir
    rights_document = args.rights.lower()
    overwrite = args.overwrite
    test = args.test

    if "openrouter" in model and providers:
        extra_body = get_openrouter_extra_body(providers, allow_fallbacks=False)
        logger.info(f"Using OpenRouter providers: {providers}")
    else:
        extra_body = None

    if eval_method == "likert":
        system_prompt = LIKERT_SYSTEM_PROMPT
        prompt_template = LIKERT_PROMPT_TEMPLATE
    elif eval_method == "open_ended":
        system_prompt = OPEN_SYSTEM_PROMPT[eval_persona]
        prompt_template = OPEN_PROMPT_TEMPLATE
    else:
        logger.error(
            f"Invalid eval_method: {eval_method}. Must be 'likert' or 'open_ended'."
        )
        return

    scenarios_file = (
        Path(working_dir) / rights_document / "scenarios" / f"{DEFAULT_LANGUAGE}.csv"
    )
    responses_file = (
        Path(working_dir)
        / rights_document
        / "responses"
        / f"{eval_persona}_{slashes_to_dashes(model)}_{language}.csv"
    )
    responses_file.parent.mkdir(parents=True, exist_ok=True)

    if not scenarios_file.exists():
        logger.error(f"Scenarios file {scenarios_file} does not exist.")
        return
    if responses_file.exists() and not overwrite:
        logger.warning(
            f"File {responses_file} already exists. Use --overwrite to overwrite."
        )
        return

    # Load the scenarios
    scenarios_df = pd.read_csv(scenarios_file)

    if test:
        logger.info("Running in test mode with limited scenarios.")
        scenarios_df = scenarios_df.head(1)
        num_samples = 1
        print(f"Test mode: Using first 5 scenarios")

    logger.info(f"Evaluating scenarios using model {model} in language {language}.")
    raw_responses_df = asyncio.run(
        run_evaluation_pipeline(
            model_id=model,
            generation_config=GenerateConfig(
                temperature=0.6,
                max_new_tokens=4000,
                max_connections=MAX_CONNECTIONS,
                max_retries=MAX_RETRIES,
                extra_body=extra_body,
            ),
            eval_method=eval_method,
            eval_persona=eval_persona,
            judge_model_id=judge_model,
            judge_generation_config=GenerateConfig(
                temperature=0,
                max_new_tokens=4000,
                max_connections=MAX_CONNECTIONS,
                max_retries=MAX_RETRIES,
            ),
            language=language,
            scenarios_df=scenarios_df,
            num_samples=num_samples,
            default_language=DEFAULT_LANGUAGE,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            open_judge_system_prompt=OPEN_JUDGE_SYSTEM_PROMPT,
            open_judge_prompt_template=OPEN_JUDGE_PROMPT_TEMPLATE,
            classification_to_score_map=CLASSIFICATION_TO_SCORE_MAP,
        )
    )

    # Save the evaluated responses
    logger.info(f"Saving evaluated responses to {responses_file}.")
    raw_responses_df.to_csv(responses_file, index=False)


if __name__ == "__main__":
    main()
