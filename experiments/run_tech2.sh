#!/usr/bin/env bash
# 技術の探索・第2弾（docs/rebuild_plan.md）を順に流す。結果が出ている実験は飛ばす。
# 使い方: bash experiments/run_tech2.sh
cd "$(dirname "$0")/.."
PY=/c/Users/aassz/anaconda3/envs/keiba-ai/python.exe
L=logs/experiments
C="--base base_pl3 --res-years 0 --half-life 730 --res-market --ref res_mkt"
run() {
    local log=$L/$1.log; shift
    if [ -f "$log" ] && grep -qE "確かめる期間|最良:|保存:" "$log"; then echo "済み: $log"; return; fi
    echo "実行: $log"; $PY -m evaluate "$@" > "$log" 2>&1
}
run tech_t8_shin    --tag t8_shin $C --shin
run tech_t9_restack --tag t9_restack $C --restack
run tech_t11_topk   --tag t11_topk $C --res-topk 60
run tech_t12_split  --tag t12_split $C --res-split
run tech_t13_adv    --tag t13_adv $C --adv-weight
echo all-done
# 第3弾（docs/rebuild_plan.md「技術の探索・第3弾」）
run tech_t15_subspace --tag t15_subspace $C --res-subspace 5
[ -f $L/tech_t14_stack.log ] || $PY -m experiments.stack_runs --ref res_mkt --tag t14_stack > $L/tech_t14_stack.log 2>&1
echo all-done-3
# T1 の続き（DART の学習回数を固定して、終わった試行は飛ばす）
run tech_t1_tune_b --tag t1_tune $C --tune 24
echo all-done-t1
