# Windows setup: install Ollama via winget (if missing) and pull the benchmark models.
# Run in PowerShell:  powershell -ExecutionPolicy Bypass -File scripts\setup_models.ps1
$ErrorActionPreference = "Stop"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements
    Write-Host "Ollama installed. Open a NEW terminal so PATH updates, then re-run this script."
    exit 0
}

try { Invoke-RestMethod http://127.0.0.1:11434/api/version | Out-Null }
catch {
    Write-Host "Starting ollama serve..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

ollama pull phi3
ollama pull mistral
ollama pull gemma2:9b

# Optional: exact GGUF quant from Hugging Face
# ollama pull hf.co/bartowski/gemma-2-9b-it-GGUF:Q4_K_M

ollama list
