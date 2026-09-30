#!/usr/bin/env bash
# 技術の探索（docs/rebuild_plan.md「技術の探索」）を順に流す。結果が出ている実験は飛ばすので、止まっても再実行で続きから。
# 使い方: bash v2/experiments/run_tech.sh
cd "$(dirname "$0")/../.."
PY=/c/Users/aassz/anaconda3/envs/keiba-ai/python.exe
L=logs/experiments
C="--base base_pl3 --res-years 0 --half-life 730 --res-market --ref res_mkt"
run() {  # run <ログ名> <引数...>
    local log=$L/$1.log; shift
    if [ -f "$log" ] && grep -qE "確かめる期間|最良:|保存:" "$log"; then echo "済み: $log"; return; fi
    echo "実行: $log"; $PY -m v2.evaluate "$@" > "$log" 2>&1
}
run tech_t3_quarter --tag t3_quarter $C --refit quarter
run tech_t1_tune    --tag t1_tune $C --tune 24
run tech_t4_build   --build-base base_pl5 --base-objective pl --k 5
run tech_t4_pl5     --tag t4_pl5 --base base_pl5 --res-years 0 --half-life 730 --res-market --ref res_mkt
echo all-done
