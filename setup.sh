#!/usr/bin/env bash
set -e

CYAN='\033[0;36m'; PURPLE='\033[0;35m'; ORANGE='\033[0;33m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

banner() {
  echo -e "${CYAN}"
  cat <<'EOF'
  ╔══════════════════════════════════════════════════════╗
  ║   LocalVideoGen Setup — Apple Silicon Edition        ║
  ╚══════════════════════════════════════════════════════╝
EOF
  echo -e "${NC}"
}

prompt_yn() {
  local msg="$1"
  read -r -p "$(echo -e ${ORANGE}?${NC}) $msg [Y/n] " ans
  [[ -z "$ans" || "$ans" =~ ^[Yy]$ ]]
}

check() {
  local name="$1"; local cmd="$2"
  if command -v "$cmd" >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} $name found ($($cmd --version 2>&1 | head -n1))"
    return 0
  fi
  echo -e "  ${RED}✗${NC} $name missing"
  return 1
}

banner

# Homebrew
if ! command -v brew >/dev/null 2>&1; then
  echo -e "${RED}Homebrew not installed.${NC}"
  if prompt_yn "Install Homebrew now?"; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  else
    echo "Aborting — Homebrew required."; exit 1
  fi
fi
echo -e "  ${GREEN}✓${NC} Homebrew"

# Node
if ! check "Node.js" node; then
  prompt_yn "Install Node 20 via brew?" && brew install node@20 || true
fi

# FFmpeg
if ! check "FFmpeg" ffmpeg; then
  prompt_yn "Install FFmpeg via brew?" && brew install ffmpeg || true
fi

# Python
if ! check "Python 3.11" python3.11; then
  prompt_yn "Install Python 3.11 via brew?" && brew install python@3.11 || true
fi

# Ollama
if ! check "Ollama" ollama; then
  prompt_yn "Install Ollama via brew?" && brew install ollama || true
fi

echo
echo -e "${PURPLE}▸ npm install${NC}"
npm install

echo
if prompt_yn "Create Python venv and install diffusers/torch (large download)?"; then
  python3.11 -m venv .venv || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install torch torchvision diffusers transformers accelerate Pillow safetensors
  deactivate
  echo -e "${GREEN}✓ Python deps installed in .venv${NC}"
  echo "  Tip: set PYTHON_BIN=$(pwd)/.venv/bin/python in .env"
fi

echo
if prompt_yn "Pull Ollama models (llava + mistral)?"; then
  if ! pgrep -x ollama >/dev/null; then
    echo "Starting ollama serve in background…"
    ollama serve >/tmp/ollama.log 2>&1 &
    sleep 3
  fi
  ollama pull llava
  ollama pull mistral
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo -e "${GREEN}✓ Created .env from .env.example${NC}"
fi

echo
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
echo
echo -e "  Next steps:"
echo -e "    ${CYAN}1.${NC} ollama serve     ${PURPLE}# in a separate terminal${NC}"
echo -e "    ${CYAN}2.${NC} npm run dev      ${PURPLE}# starts API + web UI${NC}"
echo -e "    ${CYAN}3.${NC} open http://localhost:3000"
echo
