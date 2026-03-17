#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

uvicorn backend_services.api:app --reload --app-dir . &
API_PID=$!

cleanup() {
  kill "$API_PID"
}

trap cleanup EXIT

streamlit run frontend/streamlit_app.py
