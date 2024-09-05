# Ensure your device has sufficient RAM.

# This workflow presupposes that you are using a local instance of the llama_index library (previously known as GPT Index).
# This allows interaction with document indices via models like llama or similar. 
# This setup doesn't rely on a specific instance of the LLaMA language model directly.
# However, it does expect you to have the necessary infrastructure (like VectorStoreIndex, RetrieverQueryEngine, and other components from llama_index) running locally.
# Such dependencies are needed to interact with vector-based retrieval mechanisms.

## Interacting with BioBERT:
# In this case, the llama_index (or GPT Index) library can work with vector embeddings from any model, like the BioBERT embeddings you are generating. 
# The workflow uses the embeddings produced by BioBERT to build a vector index for ClinVar data and subsequently retrieve relevant data for queries.




# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python using Homebrew
brew install python

# Confirm Python is installed
python3 --version

# Install pip (if needed)
python3 -m ensurepip --upgrade

# Create and activate a virtual environment (replace 'clinvar_env' with your desired name)
python3 -m venv clinvar_env
source clinvar_env/bin/activate

# Install required Python packages (transformers, PyTorch, llama_index, pandas)
pip3 install torch transformers pandas llama_index

# Optional: Install PyTorch with GPU (MPS) support for M1/M2 MacBook Pros
pip3 install torch torchvision torchaudio

# If using MPS (Metal Performance Shaders) for GPU on macOS:
import torch
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Optional: Install FAISS for faster vector search
brew install faiss

# To activate the virtual environment in future sessions, run:
source clinvar_env/bin/activate

# Run your Python script
python your_script.py
