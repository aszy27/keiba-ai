# scrape/__main__.py
# スクレイプの入口。プロジェクトのルートで実行する。
#
#   python -m scrape weekly                      # 毎週の取得をまとめて実行（下の results → pedigree → extras → odds → repair）
#                                                # 最近の db.netkeiba のページからは芝ダ・距離・馬場が取れないので、repair まで毎週必要
#   python -m scrape results [--year 2026]       # レース結果 → data/{train,val,test}/race_data_YYYY.csv
#   python -m scrape pedigree                    # 父・母 → data/master_horse_data.csv
#   python -m scrape extras                      # ラップ・払戻・追い切り・生産者 → data/*_progress.csv
#   python -m scrape odds [--years 2026]         # 確定の単勝・複勝オッズ → data/odds_api_progress.csv
#   python -m scrape odds-exotic --years 2020 [--types 7,8] [--limit 500]   # 組み合わせ券の確定オッズ
#   python -m scrape repair [--all] [--dry-run]  # レース情報の欠損補完・偽レースの削除
#   python -m scrape rescrape (--ids ID,... | --auto) [--year Y] [--dry-run]   # 指定レースを取り直す
#   python -m scrape snapshot [--date YYYYMMDD] [--now]   # 発走前オッズのスナップショット（前向き検証。タスクスケジューラが毎日起動）
import argparse
import datetime
from pathlib import Path


def main():
    this_year = datetime.date.today().year
    ap = argparse.ArgumentParser(prog="python -m scrape", description="netkeiba からデータを取得する")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("weekly", help="毎週の取得をまとめて実行")
    p.add_argument("--year", type=int, default=this_year)
    p = sub.add_parser("results", help="レース結果")
    p.add_argument("--year", type=int, default=this_year)
    sub.add_parser("pedigree", help="父・母")
    sub.add_parser("extras", help="ラップ・払戻・追い切り・生産者")
    p = sub.add_parser("odds", help="確定の単勝・複勝オッズ")
    p.add_argument("--years", default=str(this_year), help="カンマ区切り（例: 2024,2025,2026）")
    p = sub.add_parser("odds-exotic", help="組み合わせ券の確定オッズ")
    p.add_argument("--years", required=True)
    p.add_argument("--types", default="7,8", help="4=馬連 / 5=ワイド / 7=3連複 / 8=3連単")
    p.add_argument("--limit", type=int, help="1回の実行で取るレース数の上限")
    p.add_argument("--sleep", type=float, default=0.7)
    p = sub.add_parser("repair", help="レース情報の欠損補完・偽レースの削除")
    p.add_argument("--all", action="store_true", help="train / val / test すべてを対象にする（既定は test のみ）")
    p.add_argument("--dry-run", action="store_true", help="一覧を表示するだけ")
    p = sub.add_parser("rescrape", help="指定レースを取り直す")
    p.add_argument("--ids", help="レースID（カンマ区切り）")
    p.add_argument("--auto", action="store_true", help="抜けているレース・偽レースを自動で探す")
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="取得内容を表示するだけで保存しない")
    p = sub.add_parser("snapshot", help="発走前オッズのスナップショット")
    p.add_argument("--date", default=datetime.datetime.now().strftime("%Y%m%d"))
    p.add_argument("--minutes", default=None, help="発走の何分前に取るか（カンマ区切り。既定は snapshot.LADDER）")
    p.add_argument("--now", action="store_true", help="待たずに全レースを1回ずつ取得する（動作確認用）")
    p.add_argument("--out-dir", type=Path, default=None, help="保存先（既定は data/v2/odds_snapshots）")
    args = ap.parse_args()

    if args.cmd in ("weekly", "results"):
        from scrape import results
        results.scrape_year(args.year)
    if args.cmd in ("weekly", "pedigree"):
        from scrape import pedigree
        pedigree.run()
    if args.cmd in ("weekly", "extras"):
        from scrape import extras
        extras.run()
    if args.cmd == "weekly":
        from scrape import odds
        odds.run([str(args.year)])
    if args.cmd == "odds":
        from scrape import odds
        odds.run([y.strip() for y in args.years.split(",")])
    if args.cmd == "odds-exotic":
        from scrape import odds
        odds.run_exotic([int(y) for y in args.years.split(",")], [int(t) for t in args.types.split(",")],
                        args.limit, args.sleep)
    if args.cmd in ("weekly", "repair"):
        from scrape import repair
        repair.run(all_dirs=getattr(args, "all", False), dry_run=getattr(args, "dry_run", False))
    if args.cmd == "rescrape":
        from scrape import rescrape
        rescrape.run(args)
    if args.cmd == "snapshot":
        from scrape import snapshot
        snapshot.run(args)


if __name__ == "__main__":
    main()
