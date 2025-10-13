#! /bin/bash

echo "🚀 Starting setup process..."

# Install uv
echo "📦 Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
echo "✅ uv installed successfully"

# Install submodules
# echo "🔄 Initializing git submodules..."
# git submodule update --init --recursive
# echo "✅ Submodules initialized successfully"

# Create and activate virtual environment
echo "🌍 Creating Python virtual environment..."
uv python install 3.11
uv venv .venv
echo "  → Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment created and activated"

# Add virtual environment to Jupyter
# echo "📝 Adding virtual environment to Jupyter..."
# python -m ipykernel install --user --name=moral-lens
# echo "  → Kernel name: moral-lens"
# echo "✅ Virtual environment added to Jupyter"

# Install dependencies
echo "📚 Installing dependencies..."
echo "  → Installing project requirements..."
# uv pip install -r pyproject.toml
uv pip install -r ./requirements.txt
echo "✅ All dependencies installed"

# Setup wandb
# echo "🔑 Setting up Weights & Biases..."
# echo -n "Would you like to set up Weights & Biases? (y/n): "
# read -r SETUP_WANDB
# if [[ "$SETUP_WANDB" =~ ^[Yy]$ ]]; then
#     echo "Please get your API key from: https://wandb.ai/authorize"
#     echo -n "Enter your wandb API key: "
#     read -r WANDB_KEY
#     wandb login "$WANDB_KEY"
#     echo "✅ Successfully logged into wandb"
# else
#     echo "⏩ Skipping wandb setup"
# fi

# Setup Hugging Face
# echo "🤗 Setting up Hugging Face..."
# echo -n "Would you like to set up Hugging Face? (y/n): "
# read -r SETUP_HF
# if [[ "$SETUP_HF" =~ ^[Yy]$ ]]; then
#     echo "You'll be prompted to enter your Hugging Face token"
#     huggingface-cli login
#     echo "✅ Hugging Face setup completed"
# else
#     echo "⏩ Skipping Hugging Face setup"
# fi

echo "🎉 Setup completed successfully!"
echo "💡 Virtual environment is now created. When opening a new terminal, activate it with: source .venv/bin/activate"