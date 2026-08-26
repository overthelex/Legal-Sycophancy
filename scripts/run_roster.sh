#!/usr/bin/env bash
# Run the full frontier roster over every arm, N models at a time.
#
# Each model is a separate process with its own checkpoint directory, so a model
# that dies takes only itself down and resumes from where it stopped. Slugs are
# fully qualified on purpose: "openai/gpt-5.6" is ambiguous between terra, sol and
# luna, which differ tenfold in price, and a bare alias resolves silently.
#
#   OPENROUTER_API_KEY=... MLFLOW_TRACKING_PASSWORD=... ./scripts/run_roster.sh
#
# Env knobs: CONC (models at once, default 4), WORKERS (per model, default 20),
# SAMPLES (default 3), OUT (results dir), CASES, SUMMARIES.
set -uo pipefail

CASES=${CASES:-data/processed/livehrb_1k.json}
SUMMARIES=${SUMMARIES:-data/processed/summaries_grok46.json}
OUT=${OUT:-data/experiments/full_scale}
SAMPLES=${SAMPLES:-3}
WORKERS=${WORKERS:-20}
CONC=${CONC:-4}
LOGS=${LOGS:-logs/roster}

MODELS=(
  "openai/gpt-5.6-terra"
  "anthropic/claude-opus-4.8"
  "google/gemini-3.5-flash"
  "deepseek/deepseek-v4-pro"
  "deepseek/deepseek-v4-flash"
  "qwen/qwen3-235b-a22b"
  "qwen/qwen3-32b"
  "qwen/qwen3-8b"
)

: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-https://mlflow.lex}
export MLFLOW_TRACKING_USERNAME=${MLFLOW_TRACKING_USERNAME:-admin}
export MLFLOW_TRACKING_INSECURE_TLS=${MLFLOW_TRACKING_INSECURE_TLS:-true}
: "${MLFLOW_TRACKING_PASSWORD:?set MLFLOW_TRACKING_PASSWORD}"

for f in "$CASES" "$SUMMARIES"; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
mkdir -p "$LOGS" "$OUT"

echo "cases=$CASES summaries=$SUMMARIES samples=$SAMPLES workers=$WORKERS conc=$CONC"
echo "models: ${#MODELS[@]}   logs: $LOGS"
echo

start=$(date +%s)
for model in "${MODELS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n; done
  slug=${model//\//_}; slug=${slug//./_}
  echo "[$(date +%H:%M:%S)] start $model"
  python3 experiments/run_perturbation_openai.py \
      --cases "$CASES" --summaries "$SUMMARIES" --model "$model" \
      --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
      --samples "$SAMPLES" --workers "$WORKERS" --rq all \
      --output-dir "$OUT" > "$LOGS/$slug.log" 2>&1 &
done
wait
echo
echo "[$(date +%H:%M:%S)] roster finished in $(( ($(date +%s) - start) / 60 )) min"

# A model that failed leaves a short log and no results; say so rather than
# letting the analysis quietly run on seven models and read as eight.
for model in "${MODELS[@]}"; do
  slug=${model//\//_}; slug=${slug//./_}
  for arm in baseline rq1 rq2 rq3; do
    [ -s "$OUT/$slug/${arm}_results.json" ] || echo "MISSING $slug/$arm"
  done
done

echo
echo "next: python3 scripts/analyse_perturbation_run.py --run-dir $OUT"
