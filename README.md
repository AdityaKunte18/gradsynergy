1. Create a venv and install deps (use python3.12, assuming CUDA driver 11.8):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --index-url <your URL> torch torchvision torchaudio
   pip install -U peft transformers accelerate matplotlib numpy pandas
   ```
   