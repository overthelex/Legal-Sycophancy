import asyncio
from typing import List, Optional
from inspect_ai.model import (
    ModelUsage,
    get_model,
    GenerateConfig,
    ChatMessage,
    ChatMessageUser,
    ChatMessageSystem,
)
from dotenv import load_dotenv
from src.utils import tqdm_gather

load_dotenv()


class LLMClient:
    """Simple LLM client using UK AISI Inspect framework."""

    def __init__(self, model_id: str, config: GenerateConfig):
        """
        Initialize the LLM client.

        Args:
            model_id: Model identifier (e.g., 'openai/gpt-4.1', 'openrouter/qwen/qwen3-32b')
            config: Base GenerateConfig (set max_connections here for concurrency)
        """
        self.model_id = model_id
        self.config = config or GenerateConfig(
            max_connections=20, max_retries=3, timeout=120
        )
        self.model = get_model(model_id, config=self.config)

        self.usage: ModelUsage = ModelUsage()

    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        config: Optional[GenerateConfig] = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt/question
            system_message: Optional system message
            config: Optional configuration for generation

        Returns:
            The generated response as a string
        """
        messages: List[ChatMessage] = []

        if "qwen3" in self.model_id:
            # disable reasoning for qwen3 models
            if system_message:
                system_message += " /no_think"
            else:
                system_message = "/no_think"

        if system_message:
            messages.append(ChatMessageSystem(content=system_message))

        messages.append(ChatMessageUser(content=prompt))

        # Merge any additional config
        merged_config = self.config.merge(config) if config else self.config

        result = await self.model.generate(input=messages, config=merged_config)

        self.usage += result.usage

        return result.completion

    async def generate_batch(
        self,
        prompts: List[str],
        system_messages: Optional[List[str]] = None,
        config: Optional[GenerateConfig] = None,
        desc: str = "Generating",
        return_exceptions: bool = False,
    ) -> List[str]:
        """
        Generate responses for multiple prompts concurrently.

        Args:
            prompts: List of user prompts/questions
            system_message: Optional system message for all prompts
            config: Optional configuration for generation

        Returns:
            List of generated responses as strings
        """
        if system_messages is None:
            system_messages = [None] * len(prompts)

        tasks = [
            self.generate(prompt, system_message, config=config)
            for prompt, system_message in zip(prompts, system_messages)
        ]

        results = await tqdm_gather(*tasks, desc=desc, return_exceptions=True)

        if return_exceptions:
            # Convert exceptions to strings while preserving successes
            out: List[str] = []
            for r in results:
                if isinstance(r, Exception):
                    out.append(f"[ERROR] {type(r).__name__}: {r}")
                else:
                    out.append(r)
            return out
        else:
            # If any task failed, raise the first error (common gather behavior)
            for r in results:
                if isinstance(r, Exception):
                    raise r
            return results  # type: ignore[return-value]


# Convenience function for quick usage
async def query_llm(
    model_id: str,
    prompt: str,
    system_message: Optional[str] = None,
    config: Optional[GenerateConfig] = None,
) -> str:
    """
    Quick function to query an LLM with a single prompt.

    Args:
        model_id: Model identifier
        prompt: The user prompt
        system_message: Optional system message
            **config_kwargs: Configuration options

    Returns:
        The generated response
    """
    client = LLMClient(model_id, config=config)
    return await client.generate(prompt, system_message)


async def main():
    # Example 1: Simple single query
    print("=== Single Query Example ===")
    model_id = "openai/gpt-4"
    response = await query_llm(
        model_id, "What is 2+2?", config=GenerateConfig(temperature=0.1)
    )
    print(f"Response: {response}")

    # Example 2: Using the client directly
    print("\n=== Client Example ===")
    client = LLMClient(
        "openai/gpt-4", config=GenerateConfig(temperature=0.7, max_tokens=50)
    )
    response = await client.generate(
        "Tell me a fun fact about Python programming.",
        system_message="You are a helpful coding assistant.",
    )
    print(f"Response: {response}")

    # Example 3: Batch queries
    print("\n=== Batch Query Example ===")
    prompts = [
        "In one word, what color is the sky?",
        "What is 5 x 3?",
        "Name one planet in our solar system.",
    ]
    responses = await client.generate_batch(prompts)
    for i, response in enumerate(responses):
        print(f"Q{i + 1}: {prompts[i]}")
        print(f"A{i + 1}: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
