#!/usr/bin/env bash
# 候補3に向けた実験（docs/rebuild_plan.md「候補3に向けた実験」）。結果が出ている実験は飛ばす。
cd "$(dirname "$0")/.."
PY=/c/Users/aassz/anaconda3/envs/keiba-ai/python.exe
L=logs/experiments
R="--res-years 0 --half-life 730 --res-market"
run() {
    local log=$L/$1.log; shift
    if [ -f "$log" ] && grep -qE "確かめる期間|保存:" "$log"; then echo "済み: $log"; return; fi
    echo "実行: $log"; $PY -m evaluate "$@" > "$log" 2>&1
}
pool() {
    local log=$L/$1.log; local tag=$2; local tags=$3
    if [ -f "$log" ] && grep -q "確かめる期間" "$log"; then echo "済み: $log"; return; fi
    echo "実行: $log"; $PY -m experiments.combine_runs --tags "$tags" --ref t7_combine --tag "$tag" > "$log" 2>&1
}
# E1: base_pl5 の上の残差の版
run e1_pl5_seeds --tag pl5_seeds --base base_pl5 $R --res-seeds --ref t4_pl5
run e1_pl5_place --tag pl5_place --base base_pl5 $R --res-market-extra place --ref t4_pl5
run e1_pl5_past  --tag pl5_past  --base base_pl5 $R --past-mkt --ref t4_pl5
pool e1_pool8 e1_pool8 res_mkt,res_seeds,mkt_place,past_mkt,t4_pl5,pl5_seeds,pl5_place,pl5_past
# E2: λ=1.0 の基礎モデル
run e2_build --build-base base_pl3_l10 --base-objective pl --lam 1.0
run e2_l10_mkt --tag l10_mkt --base base_pl3_l10 $R --ref res_mkt
# E3: λ=1.0 の上の残りの版と12版のプーリング
run e3_l10_seeds --tag l10_seeds --base base_pl3_l10 $R --res-seeds --ref l10_mkt
run e3_l10_place --tag l10_place --base base_pl3_l10 $R --res-market-extra place --ref l10_mkt
run e3_l10_past  --tag l10_past  --base base_pl3_l10 $R --past-mkt --ref l10_mkt
pool e3_pool12 e3_pool12 res_mkt,res_seeds,mkt_place,past_mkt,t4_pl5,pl5_seeds,pl5_place,pl5_past,l10_mkt,l10_seeds,l10_place,l10_past
echo all-done
