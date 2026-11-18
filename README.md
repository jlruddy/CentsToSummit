# Kalshi AI Market-Impact Research POC (`CentsToSummit`)

Production-grade local Proof-of-Concept automated trading system in Python 3.12+ (managed via `uv`) for **Kalshi prediction markets**, featuring strict separation between **AI decision-making** and **deterministic risk enforcement**, full SQLite WAL + Parquet telemetry, and an integrated **Web UI Control Panel & Real-Time Activity Dashboard**.

Defaults to **Kalshi's Demo Sandbox** (`external-api.demo.kalshi.co`).

---

## 1. Quickstart (macOS Apple Silicon)

```bash
# Install dependencies and create virtual environment
uv sync

# Copy example environment file (optional; Web UI also manages config)
cp .env.example .env

# Run the unified Web UI + Trading Engine
uv run kalshi-poc run
```

Open the Control Panel at **http://127.0.0.1:8000**.

---

## 2. Key Generation Guide (Ed25519 or RSA-PSS)

Kalshi API v2 supports both **Ed25519** and **RSA-PSS** keys.

### Option A: Ed25519 (Recommended)
```bash
mkdir -p ~/.kalshi && chmod 700 ~/.kalshi
openssl genpkey -algorithm Ed25519 -out ~/.kalshi/demo_private_key.pem
openssl pkey -in ~/.kalshi/demo_private_key.pem -pubout -out ~/.kalshi/demo_public_key.pem
chmod 600 ~/.kalshi/demo_private_key.pem
```

### Option B: RSA-PSS (4096-bit)
```bash
mkdir -p ~/.kalshi && chmod 700 ~/.kalshi
openssl genrsa -out ~/.kalshi/demo_private_key.pem 4096
openssl rsa -in ~/.kalshi/demo_private_key.pem -pubout -out ~/.kalshi/demo_public_key.pem
chmod 600 ~/.kalshi/demo_private_key.pem
```

Upload the `.pem` public key in your Kalshi Demo (or Production) account settings, copy the **API Key ID**, and paste/upload your private key directly in the Web UI's **Kalshi API Credentials** card. Click **Test** to verify signature, algorithm detection, latency, and balance.

---

## 3. Web UI Control Panel Guide

- **Top Status Bar:** Shows Engine State (`RUNNING` / `PAUSED` / `KILL-SWITCH TRIPPED`), Environment (`DEMO` / `LIVE`), WebSocket health, Cash Balance, Deployable Risk Budget (`Equity - Floor`), and Session Net P&L.
- **Emergency Flatten & Halt:** Trips the kill-switch (persisted across restarts), pauses new entries, cancels resting orders (demo/live), and sends one IOC closing order per open position at the live inside quote. The response lists each position's flatten status and any remaining contracts.
- **Pause vs. kill-switch:** Both block *new entries only*. Take-profit / stop-loss exits keep running for positions opened in the mode that was active before the pause or kill. A process that boots already paused does not touch positions until you press Start. Pressing Start is the explicit operator action that clears a tripped kill-switch.
- **Live trading gate:** `execution_mode=live` is rejected by the server (UI, API, and Start) unless the process was launched with `--confirm-live`.
- **Access control:** The control panel only answers requests whose `Host` is `127.0.0.1`, `localhost`, or `::1`, and rejects cross-site writes and WebSocket connections (by `Origin`). If you serve it under another hostname, add it to `WEB_ALLOWED_HOSTS` (JSON array).
- **Hot-Reloadable Config Cards:** Edit Execution Mode (`paused`, `shadow`, `demo`, `live`), Capital & Equity Floor, Risk & Microstructure Limits, Tax-Hygiene Series Allowlist, and AI Strategy Provider (`stub_momentum`, `anthropic`, `gemini`, `ollama_local`), then click **Save & Apply** without dropping active WebSocket feeds.
- **Live Activity Stream:** Inspect derived YES/NO orderbook depth, filterable `AI_DECISION` / `RISK_GATE` / `EXECUTION` cards, and the live equity vs. `EQUITY_FLOOR` chart.

---

## 4. CLI Commands

```bash
# Start Web UI + Trading Engine (default: 127.0.0.1:8000)
uv run kalshi-poc run

# Start with explicit host/port or allow live production mode
uv run kalshi-poc run --host 127.0.0.1 --port 8000 --confirm-live

# Export all SQLite WAL telemetry tables (full history) to Parquet:
# orderbook_snapshots, ai_decisions, llm_calls, risk_evaluations, orders_and_fills, equity_curve, exchange_events
uv run kalshi-poc export --format parquet --output-dir data/exports

# Run test suite
uv run pytest
```

---

## 5. Docker Deployment to AWS EC2 (`us-east-2`)

```bash
# On your EC2 instance (with Docker & Docker Compose installed):
cp .env.example .env
# Populate .env and place private key in ~/.kalshi/
docker compose up -d --build
```

Use an SSH tunnel from your local machine to access the control panel securely without exposing port 8000 publicly:
```bash
ssh -N -L 8000:127.0.0.1:8000 ec2-user@<EC2_PUBLIC_IP>
```
