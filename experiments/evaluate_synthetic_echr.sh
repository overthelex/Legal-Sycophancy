#!/bin/bash


# Define run parameters
models=(
  "openrouter/qwen/qwen3-235b-a22b|together/fp8 deepinfra/fp8"
  # "openrouter/qwen/qwen-2.5-72b-instruct|nebius/fp8"

  # "openrouter/meta-llama/llama-4-maverick|deepinfra/base novita/fp8"
  # "openrouter/meta-llama/llama-3.3-70b-instruct|nebius/fp8 deepinfra/turbo"

  # "openrouter/google/gemma-3-27b-it|nebius/fp8 deepinfra/fp8"

  # "openrouter/mistralai/mistral-medium-3|mistral"
  # "openrouter/mistralai/mistral-large-2407|mistral"

  # "openrouter/deepseek/deepseek-chat-v3-0324|novita/fp8 nebius/fp8"

  # "openrouter/openai/gpt-3.5-turbo|openai"
  # "openrouter/openai/gpt-4o-2024-11-20|openai"

  # "openrouter/anthropic/claude-sonnet-4|google-vertex google-vertex/europe anthropic"


  # Scale experiments #
  # "openrouter/qwen/qwen3-32b|nebius/base"
  # "openrouter/qwen/qwen3-14b|nebius/fp8"
  # "openrouter/qwen/qwen3-8b|novita/fp8"

  # "openrouter/google/gemma-3-12b-it|deepinfra/bf16 novita/bf16"
  # "openrouter/google/gemma-3-4b-it|deepinfra/bf16"

  # "openrouter/mistralai/mistral-small|mistral"
  # "openrouter/mistralai/mistral-nemo|mistral"
)

languages=(
  "en"
  # "zh-cn"
  # "ro"
  # "es"
  # "ru"
  # "ar"
  # "zu"
  # "hi"
)

personas=(
  "none"
  # "individual-rights"
  # "government-power"
)

# Loop over combinations
for tup in "${models[@]}"; do
  if [[ "$tup" == *"|"* ]]; then
    model="${tup%%|*}"
    providers_str="${tup#*|}"
    # split by whitespace into an array
    IFS=' ' read -r -a providers_arr <<< "$providers_str"
  else
    model="$tup"
    providers_arr=()
  fi

  for lang in "${languages[@]}"; do
    for persona in "${personas[@]}"; do
      echo "🚀 Running with model=$model, lang=$lang, persona=$persona, providers=${providers_arr[*]}"

      # build common args in an array to keep word-safety
      args=(
        -m src.synthetic.evaluation
        --language "$lang"
        --model "$model"
        --eval_persona "$persona"
        --rights "echr"
        # --overwrite
        # --test
      )

      # only include --providers when we actually have any
      if ((${#providers_arr[@]})); then
        python "${args[@]}" --providers "${providers_arr[@]}"
      else
        python "${args[@]}"
      fi
    done
  done
done
