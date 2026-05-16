#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Background da serverni ishga tushirish
# ════════════════════════════════════════════════════════════════════════════
set -e

cd "$(dirname "$0")"

# Eski jarayonni o'chir
if [ -f app.pid ]; then
    OLD_PID=$(cat app.pid)
    if kill -0 $OLD_PID 2>/dev/null; then
        echo "Eski jarayon to'xtatilmoqda (PID: $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    fi
    rm -f app.pid
fi

# Ollama ishlamayotgan bo'lsa, ishga tushir
if ! curl -s http://localhost:11435/api/tags > /dev/null 2>&1; then
    echo "Ollama ishga tushirilmoqda..."
    sudo systemctl start ollama 2>/dev/null || nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 3
fi

# Venv ni aktivlashtir va serverni ishga tushir
source .venv/bin/activate

echo "Server ishga tushirilmoqda..."
nohup python3 app.py > app.log 2>&1 &
echo $! > app.pid

sleep 2

if kill -0 $(cat app.pid) 2>/dev/null; then
    echo "✅ Server ishlamoqda (PID: $(cat app.pid))"
    echo "   URL : http://$(hostname -I | awk '{print $1}'):8000"
    echo "   Log : tail -f app.log"
    echo "   Stop: bash stop.sh"
else
    echo "❌ Server ishga tushmadi. Log:"
    tail -20 app.log
fi
