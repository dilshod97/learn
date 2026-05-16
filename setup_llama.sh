#!/bin/bash
# llama.cpp o'rnatish (GGUF conversion uchun)
set -e

echo "📦 llama.cpp yuklab olinmoqda..."
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

echo "🔨 Build qilinmoqda (CUDA bilan)..."
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)

echo "📦 Python paketlari..."
pip install -r requirements/requirements-convert_hf_to_gguf.txt

echo "✅ llama.cpp tayyor!"
