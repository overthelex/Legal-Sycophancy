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
# Env knobs: CONC (models at once, default 8), WORKERS_OVERRIDE (flat worker count,
# overriding the calibrated per-model values), RQ (arms, default all), SAMPLES,
# OUT, CASES, SUMMARIES.
set -uo pipefail

CASES=${CASES:-data/processed/livehrb_1k.json}
SUMMARIES=${SUMMARIES:-data/processed/summaries_grok46.json}
OUT=${OUT:-data/experiments/full_scale}
SAMPLES=${SAMPLES:-3}
CONC=${CONC:-8}   # all eight at once; the gate exists for smaller reruns
# Which arms to run. Defaulting to "all" wasted a run: pointing --summaries at the
# extractive file and leaving this alone sent every model on to RQ3, which reads the
# full case text and not the summaries at all -- an exact duplicate of work already
# paid for, and the most expensive arm of the four.
RQ=${RQ:-all}
LOGS=${LOGS:-logs/roster}

# Workers per model, sized from measured single-call latency so every model finishes
# at roughly the same time. A flat worker count would leave the roster waiting on
# qwen3-32b, whose calls take 117s against 20s for Opus -- nearly six times longer.
# Calibrated 26 Aug on 5 cases per model at 40 total concurrent workers:
#   qwen3-32b 117s | qwen3-235b 47s | v4-pro 43s | qwen3-8b 39s
#   v4-flash 27s | gpt-5.6-terra 24s | gemini-3.5-flash 21s | opus-4.8 20s
MODELS=(
  "qwen/qwen3-32b:69"
  "qwen/qwen3-235b-a22b:28"
  "deepseek/deepseek-v4-pro:25"
  "qwen/qwen3-8b:23"
  "deepseek/deepseek-v4-flash:16"
  "openai/gpt-5.6-terra:14"
  "google/gemini-3.5-flash:12"
  "anthropic/claude-opus-4.8:12"
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

echo "cases=$CASES summaries=$SUMMARIES samples=$SAMPLES conc=$CONC rq=$RQ"
echo "models: ${#MODELS[@]}   logs: $LOGS"
echo

start=$(date +%s)
for entry in "${MODELS[@]}"; do
  model=${entry%:*}; w=${entry##*:}
  [ -n "${WORKERS_OVERRIDE:-}" ] && w=$WORKERS_OVERRIDE
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do wait -n; done
  slug=${model//\//_}; slug=${slug//./_}
  echo "[$(date +%H:%M:%S)] start $model with $w workers"
  python3 experiments/run_perturbation_openai.py \
      --cases "$CASES" --summaries "$SUMMARIES" --model "$model" \
      --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
      --samples "$SAMPLES" --workers "$w" --rq "$RQ" \
      --output-dir "$OUT" > "$LOGS/$slug.log" 2>&1 &
done
wait
echo
echo "[$(date +%H:%M:%S)] roster finished in $(( ($(date +%s) - start) / 60 )) min"

# A model that failed leaves a short log and no results; say so rather than
# letting the analysis quietly run on seven models and read as eight.
for entry in "${MODELS[@]}"; do
  model=${entry%:*}
  slug=${model//\//_}; slug=${slug//./_}
  for arm in $([ "$RQ" = all ] && echo baseline rq1 rq2 rq3 || echo "$RQ"); do
    [ -s "$OUT/$slug/${arm}_results.json" ] || echo "MISSING $slug/$arm"
  done
done

echo
echo "next: python3 scripts/analyse_perturbation_run.py --run-dir $OUT"
