#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ ! -f app.pid ]; then
    echo "app.pid topilmadi"
    exit 0
fi

PID=$(cat app.pid)
if kill -0 $PID 2>/dev/null; then
    kill $PID
    echo "✅ Server to'xtatildi (PID: $PID)"
else
    echo "Jarayon allaqachon to'xtagan"
fi
rm -f app.pid
