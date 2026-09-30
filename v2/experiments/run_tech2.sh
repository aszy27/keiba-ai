#!/usr/bin/env bash
# 技術の探索・第2弾（docs/rebuild_plan.md）を順に流す。結果が出ている実験は飛ばす。
# 使い方: bash v2/experiments/run_tech2.sh
cd "$(dirname "$0")/../.."
PY=/c/Users/aassz/anaconda3/envs/keiba-ai/python.exe
L=logs/experiments
C="--base base_pl3 --res-years 0 --half-life 730 --res-market --ref res_mkt"
run() {
    local log=$L/$1.log; shift
    if [ -f "$log" ] && grep -qE "確かめる期間|最良:|保存:" "$log"; then echo "済み: $log"; return; fi
    echo "実行: $log"; $PY -m v2.evaluate "$@" > "$log" 2>&1
}
run tech_t8_shin    --tag t8_shin $C --shin
run tech_t9_restack --tag t9_restack $C --restack
run tech_t11_topk   --tag t11_topk $C --res-topk 60
run tech_t12_split  --tag t12_split $C --res-split
run tech_t13_adv    --tag t13_adv $C --adv-weight
echo all-done
