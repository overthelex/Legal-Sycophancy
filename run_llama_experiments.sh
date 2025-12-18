#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --job-name=llm_human_rights_llama
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G

# Hugging Face cache configuration - download to scratch
export HF_HOME=~/scratch/hf_cache
export TRANSFORMERS_CACHE=~/scratch/hf_cache
export HF_DATASETS_CACHE=~/scratch/hf_cache/datasets
export HF_HUB_DOWNLOAD_TIMEOUT=120

# Load modules
module load gcc arrow

# Activate virtual environment
source .venv/bin/activate

# Navigate to project directory
cd /project/6105522/junkais/LLM-Human-Rights

# Install requirements (including Ray and vLLM)
echo "Installing requirements..."
pip install -q -r requirements.txt
pip install -q "ray[data]>=2.44.1" vllm

# Create output directory
mkdir -p slurm/output
mkdir -p data/experiments

# Set model ID
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

echo "=========================================="
echo "Starting Llama-3.1-8B Experiments"
echo "Model: $MODEL_ID"
echo "=========================================="

# Run experiments with vLLM batch inference
# You can specify which scenarios to run with --scenarios flag
# Examples:
#   --scenarios generic_generic male_generic female_generic
#   --scenarios generic_russia generic_turkey generic_ukraine

python run_unified_experiment.py \
    --use-vllm \
    --model "$MODEL_ID" \
    --scenarios generic_generic male_generic female_generic \
    --temperature 0.0 \
    --force

echo "=========================================="
echo "Experiments completed!"
echo "Results saved to: data/experiments/llama-3.1-8b-instruct/"
echo "=========================================="
