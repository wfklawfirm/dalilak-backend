#!/bin/bash
# للتشغيل المحلي فقط — ضع مفاتيحك في .env
cd "$(dirname "$0")"

if [ -f .env ]; then
    export $(cat .env | xargs)
fi

PY=""
for v in python3.12 python3.11 python3.10; do
    if command -v $v &>/dev/null; then PY=$v; break; fi
done

if [ ! -d "venv" ]; then
    $PY -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt -q
python main.py
