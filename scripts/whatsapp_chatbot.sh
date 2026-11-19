#!/bin/bash
# Start the WhatsApp (Twilio) webhook server on port 8000.
# Expose it over public HTTPS and point your Twilio WhatsApp webhook at /hook (see README).
# All keys and settings come from src/config.py (.env or AWS Secrets Manager).
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/src/server"

# --proxy-headers lets signature validation see the original https URL behind a reverse proxy.
exec uvicorn twilio_server:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips="*"
