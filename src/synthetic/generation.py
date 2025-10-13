""" """

import argparse
import asyncio
import itertools
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.asyncio import tqdm as tqdm_asyncio

from src.synthetic.generation_prompts import (
    SCENARIO_SYSTEM_PROMPT,
    SCENARIO_PROMPT_TEMPLATE,
    # TODO
    # EVALUATION_SYSTEM_PROMPT,
    # EVALUATION_PROMPT_TEMPLATE,
    # SEVERITY_EVALUATION_SYSTEM_PROMPT,
    # SEVERITY_EVALUATION_PROMPT_TEMPLATE,
    # RIGHTS_IDENTIFICATION_SYSTEM_PROMPT,
    # RIGHTS_IDENTIFICATION_PROMPT_TEMPLATE
)

from src.synthetic.config import (
    MAX_CONNECTIONS,
    MAX_RETRIES,
    DEFAULT_LANGUAGE,
    UDHR_LIMITED_ARTICLES,
    UDHR_ARTICLE_TEXT_FILE,
    ECHR_LIMITED_ARTICLES,
    ECHR_ARTICLE_TEXT_FILE,
    AFFECTED_GROUPS,
    ACTORS,
    SCENARIO_CONDITIONS,
)
from src.client import LLMClient
from inspect_ai.model import GenerateConfig
import logging

from src.utils import _log_token_usage

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


def evaluate_scenarios(
    save_file: Path,
):
    # TODO
    pass


def _count_scenarios(rights_document: str) -> int:
    """Count the number of scenarios in the current configuration."""
    combos = configure_scenarios(rights_document=rights_document)
    return len(combos) * len(AFFECTED_GROUPS)


def configure_scenarios(rights_document: str) -> List[Dict[str, Any]]:
    if rights_document == "udhr":
        file_path = UDHR_ARTICLE_TEXT_FILE
        limited_articles = UDHR_LIMITED_ARTICLES
    elif rights_document == "echr":
        file_path = ECHR_ARTICLE_TEXT_FILE
        limited_articles = ECHR_LIMITED_ARTICLES
    else:
        raise ValueError(f"Unknown rights document: {rights_document}")

    df = pd.read_csv(file_path)

    # filter to only limited articles
    df = df[df["article_num"].isin(limited_articles.keys())]

    combos: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        for actor in ACTORS:
            for severity, state_of_emergency in SCENARIO_CONDITIONS:
                combos.append(
                    {
                        "article_number": row["article_num"],
                        "article_name": row["article_name"],
                        "article_text": row["article_text"],
                        "actor": actor,
                        "affected_groups": AFFECTED_GROUPS,
                        "severity": severity,
                        "state_of_emergency": state_of_emergency,
                    }
                )
    return combos


async def run_generation_pipeline(
    model_id: str,
    generation_config: GenerateConfig,
    rights_document: str,
):
    """Generate scenarios using the specified model and translate them into the target language."""

    # Initialize clients
    client = LLMClient(model_id=model_id, config=generation_config)

    # Configure scenarios
    combos = configure_scenarios(rights_document)

    # Prepare batch generation data
    batch_prompts = []
    batch_system_messages = []

    for combo in combos:
        formatted_prompt = SCENARIO_PROMPT_TEMPLATE.format(
            limited_right=combo["article_text"],
            actor=combo["actor"],
            severity=combo["severity"],
            state_of_emergency=combo["state_of_emergency"],
        )

        batch_prompts.append(formatted_prompt)
        batch_system_messages.append(SCENARIO_SYSTEM_PROMPT)

    # Generate scenarios in batch
    logger.info(f"Generating {len(batch_prompts)} scenario templates in batch...")
    scenario_templates = await client.generate_batch(
        prompts=batch_prompts,
        system_messages=batch_system_messages,
        desc="Generating scenario templates",
    )

    # Create individual scenarios from templates
    entries = []
    id = 0
    for _, (scenario_template, metadata) in enumerate(zip(scenario_templates, combos)):
        article_number = metadata["article_number"]
        article_name = metadata["article_name"]
        article_text = metadata["article_text"]
        actor = metadata["actor"]
        affected_groups = metadata["affected_groups"]
        severity = metadata["severity"]
        state_of_emergency = metadata["state_of_emergency"]

        for affected_group in affected_groups:
            # Replace "[STATE]" and "[APPLICANT]" in the scenario template
            scenario_final = scenario_template.replace("<people>", affected_group)

            entries.append(
                {
                    "limited_article": article_number,
                    "limited_article_name": article_name,
                    "limited_article_text": article_text,
                    "actor": actor,
                    "affected_group": affected_group,
                    "severity": severity,
                    "state_of_emergency": state_of_emergency,
                    "scenario_id": id,
                    "scenario_text": scenario_final,
                }
            )
            id += 1
    df = pd.DataFrame(entries)

    usage_logfile = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_synthetic_gen_usage.log"
    )
    _log_token_usage(client.usage, usage_logfile, model_id)

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic human rights scenarios based on the UDHR."
    )
    parser.add_argument(
        "--count_scenarios",
        action="store_true",
        help="Display the number of scenarios in the current configuration.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model identifier (e.g., 'openai/gpt-4.1-2025-04-14').",
    )
    parser.add_argument(
        "--evaluate_scenario_quality",
        action="store_true",
        help="Evaluate the quality of generated scenarios.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing scenario files."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/experiments/synthetic",
        help="Directory to save generated scenarios (default: 'data/experiments/synthetic').",
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
    out_dir = args.out_dir
    evaluate_scenario_quality = args.evaluate_scenario_quality
    count_scenarios = args.count_scenarios
    overwrite = args.overwrite
    test = args.test
    rights = args.rights.lower()

    if test:
        logger.info("Running in test mode with limited scenarios.")
        global ACTORS, AFFECTED_GROUPS, SCENARIO_CONDITIONS
        ACTORS = ACTORS[:2]
        AFFECTED_GROUPS = AFFECTED_GROUPS[:2]
        SCENARIO_CONDITIONS = SCENARIO_CONDITIONS[:2]
        print(
            f"Test mode: Using ACTORS={ACTORS}, AFFECTED_GROUPS={AFFECTED_GROUPS}, SCENARIO_CONDITIONS={SCENARIO_CONDITIONS}"
        )

    save_file = Path(out_dir) / rights / "scenarios" / f"{DEFAULT_LANGUAGE}.csv"
    save_file.parent.mkdir(parents=True, exist_ok=True)

    if count_scenarios:
        num_scenarios = _count_scenarios(rights)
        logger.info(
            f"Number of scenarios in the current configuration: {num_scenarios}"
        )
        return

    if save_file.exists() and not overwrite and not evaluate_scenario_quality:
        logger.warning(
            f"File {save_file} already exists. Use --overwrite to overwrite."
        )
        return

    if not save_file.exists() or overwrite:
        logger.info(f"Running generation pipeline.")
        df_out = asyncio.run(
            run_generation_pipeline(
                model_id=model,
                generation_config=GenerateConfig(
                    temperature=0.2,
                    max_tokens=4000,
                    max_connections=MAX_CONNECTIONS,
                    max_retries=MAX_RETRIES,
                ),
                # language=language,
                rights_document=rights,
            )
        )

        logger.info(f"Saving generated scenarios to {save_file}.")
        df_out.to_csv(save_file, index=False)

    if evaluate_scenario_quality:
        evaluate_scenarios(
            save_file=save_file,
        )


if __name__ == "__main__":
    main()
