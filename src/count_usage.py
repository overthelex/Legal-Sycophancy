import os
import glob
import re
import argparse


def count_usage(logs_dir: str = "logs/token_usage") -> None:
    """Count and display accumulated token usage from log files."""

    if not os.path.exists(logs_dir):
        print(f"Directory {logs_dir} does not exist")
        return

    # Dictionary to store accumulated usage per model
    model_usage = {}

    # Pattern to extract usage information
    model_pattern = r'Model: "([^"]+)"'
    prompt_pattern = r"Total Prompt Tokens: (\d+)"
    completion_pattern = r"Total Completion Tokens: (\d+)"
    total_pattern = r"Total Tokens: (\d+)"
    cache_write_pattern = r"Cache Write: (\d+)"
    cache_read_pattern = r"Cache Read: (\d+)"
    reasoning_pattern = r"Reasoning Tokens: (\d+)"

    # Process all log files
    for log_file in glob.glob(os.path.join(logs_dir, "**", "*.log"), recursive=True):
        with open(log_file, "r") as f:
            content = f.read()

        # Split content by model entries
        entries = content.split("=" * 50)

        for entry in entries:
            if not entry.strip():
                continue

            model_match = re.search(model_pattern, entry)
            if not model_match:
                continue

            model = model_match.group(1)

            # Extract usage numbers
            prompt_tokens = (
                int(re.search(prompt_pattern, entry).group(1))
                if re.search(prompt_pattern, entry)
                else 0
            )
            completion_tokens = (
                int(re.search(completion_pattern, entry).group(1))
                if re.search(completion_pattern, entry)
                else 0
            )
            total_tokens = (
                int(re.search(total_pattern, entry).group(1))
                if re.search(total_pattern, entry)
                else 0
            )
            cache_write = (
                int(re.search(cache_write_pattern, entry).group(1))
                if re.search(cache_write_pattern, entry)
                else 0
            )
            cache_read = (
                int(re.search(cache_read_pattern, entry).group(1))
                if re.search(cache_read_pattern, entry)
                else 0
            )
            reasoning_tokens = (
                int(re.search(reasoning_pattern, entry).group(1))
                if re.search(reasoning_pattern, entry)
                else 0
            )

            # Accumulate usage
            if model not in model_usage:
                model_usage[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cache_write": 0,
                    "cache_read": 0,
                    "reasoning_tokens": 0,
                }

            model_usage[model]["prompt_tokens"] += prompt_tokens
            model_usage[model]["completion_tokens"] += completion_tokens
            model_usage[model]["total_tokens"] += total_tokens
            model_usage[model]["cache_write"] += cache_write
            model_usage[model]["cache_read"] += cache_read
            model_usage[model]["reasoning_tokens"] += reasoning_tokens

    # Display results
    print("\n🧮 Accumulated Token Usage by Model:")
    print("=" * 60)

    grand_total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_write": 0,
        "cache_read": 0,
        "reasoning_tokens": 0,
    }

    for model, usage in sorted(model_usage.items()):
        print(f"\nModel: {model}")
        print(f"  Prompt Tokens: {usage['prompt_tokens']:,}")
        print(f"  Completion Tokens: {usage['completion_tokens']:,}")
        print(f"  Total Tokens: {usage['total_tokens']:,}")
        print(f"  Cache Write: {usage['cache_write']:,}")
        print(f"  Cache Read: {usage['cache_read']:,}")
        print(f"  Reasoning Tokens: {usage['reasoning_tokens']:,}")

        # Add to grand total
        for key in grand_total:
            grand_total[key] += usage[key]

    print("\n" + "=" * 60)
    print("GRAND TOTAL:")
    print(f"  Prompt Tokens: {grand_total['prompt_tokens']:,}")
    print(f"  Completion Tokens: {grand_total['completion_tokens']:,}")
    print(f"  Total Tokens: {grand_total['total_tokens']:,}")
    print(f"  Cache Write: {grand_total['cache_write']:,}")
    print(f"  Cache Read: {grand_total['cache_read']:,}")
    print(f"  Reasoning Tokens: {grand_total['reasoning_tokens']:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count and display accumulated token usage from log files."
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/token_usage",
        help="Directory containing log files (default: logs/token_usage)",
    )
    args = parser.parse_args()
    count_usage()
