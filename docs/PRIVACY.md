# Privacy verification procedure

The claim: **document text, questions and answers never leave the machine.**

## 1. Application layer (automated)

`python -m benchmark run` installs an egress guard (`src/benchmark/egress.py`) before doing anything else.
Every `socket.connect` / `socket.create_connection` from the Python process to a non-loopback address
raises `EgressBlocked`. Loopback (`127.0.0.1`, `::1`, `localhost`) is allowed so the process can reach Ollama.

CI proves it on every push (`tests/test_e2e_fake_ollama.py::test_privacy_no_egress_during_full_run`):
with the guard on, connections to public IPs fail and a full benchmark run still completes.

`OLLAMA_HOST` pointing at a non-loopback address is also refused (`Settings.assert_offline`)
unless `ASSISTANT_ALLOW_REMOTE=1` is set deliberately.

## 2. Model server (manual, OS level)

The guard covers this application, not Ollama's own binary. To check the whole stack:

**Simplest:** pull the models, then turn on airplane mode / unplug the network and run

```powershell
python -m benchmark run --models phi3 --limit 5 --run-id offline-check
```

It must complete with 15 records in `results/offline-check/records.jsonl`.

**Stricter (Windows firewall, network stays up for everything else):**

```powershell
# Admin PowerShell: block all outbound traffic from Ollama, run, then remove the rule
New-NetFirewallRule -DisplayName "Block Ollama outbound" -Direction Outbound -Program "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -Action Block
python -m benchmark run --models phi3 --limit 5 --run-id offline-check
Remove-NetFirewallRule -DisplayName "Block Ollama outbound"
```

Loopback traffic is not affected by the rule, so the run must still succeed.

## What this does not cover

- `ollama pull` needs the internet once, to download weights.
- The Ollama desktop app may check for updates; that traffic contains no document data, and the firewall rule above blocks it.
