#!/usr/bin/env bash
cd "$(dirname "$0")"

# 1. app.pid orqali asosiy jarayonni o'ldir
if [ -f app.pid ]; then
    PID=$(cat app.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID 2>/dev/null
        sleep 1
        kill -9 $PID 2>/dev/null
        echo "✅ Asosiy jarayon to'xtatildi (PID: $PID)"
    fi
    rm -f app.pid
fi

# 2. app.py ishlatayotgan barcha qoldiq jarayonlarni topib o'ldir
LEFTOVERS=$(pgrep -f "python3 app.py" || true)
if [ -n "$LEFTOVERS" ]; then
    echo "⚠️  Qoldiq jarayonlar topildi, o'ldirilmoqda: $LEFTOVERS"
    echo "$LEFTOVERS" | xargs kill -9 2>/dev/null
fi

# 3. Training jarayonlari (CUDA da osilib qolgan)
TRAINERS=$(pgrep -f "trainer.py" || true)
if [ -n "$TRAINERS" ]; then
    echo "⚠️  Training jarayonlari: $TRAINERS"
    echo "$TRAINERS" | xargs kill -9 2>/dev/null
fi

sleep 1
echo
echo "GPU holati:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv 2>/dev/null || true
