#!/usr/bin/env bash
# One-command start for Linux/macOS: creates .venv, installs, configures, launches.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  echo "[riko] creating virtual environment (.venv)"
  "$PY" -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip >/dev/null
  echo "[riko] installing Riko Next (this can take a few minutes)"
  ./.venv/bin/pip install -e ".[all]"
fi
[ -f config.yaml ] || { cp config.example.yaml config.yaml; echo "[riko] created config.yaml"; }
[ -f .env ] || { cp .env.example .env; echo "[riko] created .env (add API keys there if you use cloud providers)"; }
GPT_SOVITS_DIR="$(grep -E '^GPT_SOVITS_DIR=' .env | cut -d= -f2- || true)"
if [ -n "${GPT_SOVITS_DIR}" ] && [ -f "${GPT_SOVITS_DIR}/api_v2.py" ]; then
  echo "[riko] starting GPT-SoVITS API from ${GPT_SOVITS_DIR}"
  ( cd "${GPT_SOVITS_DIR}" && "${GPT_SOVITS_PYTHON:-python}" api_v2.py -a 127.0.0.1 -p 9880 \
      -c GPT_SoVITS/configs/tts_infer.yaml > "$OLDPWD/data_gpt_sovits.log" 2>&1 & )
fi
exec ./.venv/bin/python -m riko serve --open "$@"
