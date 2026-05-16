#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Ubuntu Server — bir martalik to'liq sozlash
# ════════════════════════════════════════════════════════════════════════════
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn(){ echo -e "${YELLOW}⚠️  $1${NC}"; }
err() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# ── 1. Tizim paketlari ─────────────────────────────────────────────────────
log "Tizim paketlari yangilanmoqda..."
sudo apt-get update -y
sudo apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    git build-essential cmake curl wget \
    pkg-config libssl-dev

# ── 2. CUDA tekshirish ─────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    log "GPU topildi:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    warn "nvidia-smi topilmadi! Fine-tuning juda sekin bo'ladi."
    warn "CUDA o'rnatish: https://developer.nvidia.com/cuda-downloads"
fi

# ── 3. Python venv ─────────────────────────────────────────────────────────
log "Python virtual environment yaratilmoqda..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# ── 4. PyTorch (CUDA bilan) ────────────────────────────────────────────────
log "PyTorch (CUDA 12.1) o'rnatilmoqda..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# ── 5. Python paketlari ────────────────────────────────────────────────────
log "Python paketlari o'rnatilmoqda..."
pip install -r requirements.txt

# gpt-oss MXFP4 uchun triton kernels (rasmiy yo'l)
log "Triton kernels (gpt-oss MXFP4 uchun)..."
pip install -U "transformers>=4.45" "accelerate>=0.34" || true
pip install "git+https://github.com/triton-lang/triton.git#subdirectory=python/triton_kernels" || warn "triton_kernels o'rnatilmadi (gpt-oss MXFP4 uchun)"

# ── 6. Ollama ──────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    log "Ollama o'rnatilmoqda..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    log "Ollama allaqachon o'rnatilgan: $(ollama --version)"
fi

# Ollama servisini ishga tushir
sudo systemctl enable ollama 2>/dev/null || true
sudo systemctl start ollama 2>/dev/null || (nohup ollama serve > /tmp/ollama.log 2>&1 &)
sleep 3

# Asosiy modellarni yuklab olish
log "gpt-oss:20b yuklab olinmoqda (~13GB)..."
ollama pull gpt-oss:20b || warn "gpt-oss:20b yuklab olinmadi"

log "nomic-embed-text yuklab olinmoqda..."
ollama pull nomic-embed-text || true

# ── 7. llama.cpp (GGUF conversion uchun) ───────────────────────────────────
if [ ! -d "llama.cpp" ]; then
    log "llama.cpp clone qilinmoqda..."
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp

    log "llama.cpp build qilinmoqda (CUDA bilan)..."
    cmake -B build -DGGML_CUDA=ON
    cmake --build build --config Release -j$(nproc)

    # Conversion script uchun paketlar
    pip install -r requirements/requirements-convert_hf_to_gguf.txt

    # Quantize binarini PATH ga tushiramiz
    ln -sf $(pwd)/build/bin/llama-quantize ./llama-quantize 2>/dev/null || true
    cd ..
    log "✅ llama.cpp tayyor"
else
    log "llama.cpp allaqachon mavjud"
fi

# ── 8. Firewall (8000 port) ────────────────────────────────────────────────
if command -v ufw &>/dev/null; then
    sudo ufw allow 8000/tcp 2>/dev/null || true
    log "Port 8000 ochildi (ufw)"
fi

# ── 9. Tugallandi ──────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ O'rnatish tugallandi!${NC}"
echo "════════════════════════════════════════════════════════════"
echo
echo "Ishga tushirish:"
echo "   bash run.sh                    # background da"
echo "   source .venv/bin/activate && python3 app.py   # interaktiv"
echo
echo "Brauzerda: http://<server-ip>:8000"
echo "Login: admin / admin"
echo
