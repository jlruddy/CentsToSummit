#!/bin/zsh
for t in $(sqlite3 "$1" "select distinct ticker from orders_and_fills"); do curl -s "https://external-api.kalshi.com/trade-api/v2/markets/$t" | python3 -c "import sys,json; m=json.load(sys.stdin).get('market',{}); print(m.get('ticker'), m.get('status'), m.get('result') or '-', m.get('close_time'), m.get('expiration_value'))"; done
