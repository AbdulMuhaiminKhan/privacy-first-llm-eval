#!/usr/bin/env bash
# Install Ollama (Linux/macOS) and pull the three benchmark models.
# Windows: use scripts/setup_models.ps1 instead.
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  case "$(uname -s)" in
    Linux)  curl -fsSL https://ollama.com/install.sh | sh ;;
    Darwin) echo "Install the macOS app from https://ollama.com/download (or: brew install ollama)"; exit 1 ;;
    *)      echo "Unsupported OS"; exit 1 ;;
  esac
fi

# Make sure the server is up (the desktop app / systemd service normally starts it).
if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "Starting ollama serve in the background..."
  (ollama serve >/tmp/ollama.log 2>&1 &)
  sleep 3
fi

# --- The three benchmark models (Ollama library tags; all are 4-bit GGUF builds) ---
ollama pull phi3          # Phi-3 Mini 3.8B
ollama pull mistral       # Mistral 7B Instruct
ollama pull gemma2:9b     # Gemma 2 9B Instruct

# --- Optional: pull a GGUF straight from Hugging Face ---
# Pins the exact quantisation (Q4_K_M) instead of relying on the library default.
# ollama pull hf.co/bartowski/gemma-2-9b-it-GGUF:Q4_K_M

ollama list
echo "Done. Everything below this point runs fully offline."
