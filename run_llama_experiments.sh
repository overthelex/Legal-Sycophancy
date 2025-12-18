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

# Load modules (load opencv BEFORE activating venv to avoid opencv-python conflict)
module load gcc arrow opencv/4.8.1

# Activate virtual environment
source .venv/bin/activate

# Navigate to project directory
cd /project/6105522/junkais/LLM-Human-Rights

# Install requirements (excluding opencv-python since we use the module)
echo "Installing requirements..."
# Install base requirements (skip opencv-python)
pip install -q pandas numpy scipy matplotlib seaborn openai python-dotenv

# Install Ray and vLLM (critical for batch inference)
echo "Installing Ray and vLLM..."
pip install --no-cache-dir "ray[data]>=2.44.1" vllm

# Verify installations
echo "Verifying installations..."
python -c "import ray; print(f'Ray version: {ray.__version__}')" || echo "ERROR: Ray not installed"
python -c "import vllm; print(f'vLLM installed')" || echo "ERROR: vLLM not installed"

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
