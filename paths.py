# paths.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY_DIR = ROOT / "data"          # スクレイパーが書く CSV（ここからは書き換えない。修正は data/v2/patches/ に置く）
V2_DIR = LEGACY_DIR / "v2"          # v2 の表（parquet）と検査レポート
PATCH_DIR = V2_DIR / "patches"      # 旧 CSV に当てる修正（取り直したレース情報・レース丸ごとの差し替え・削除）

TABLES = ("races", "runners", "payouts", "odds_final", "laps", "training", "horses", "courses")


def table_path(name):
    return V2_DIR / f"{name}.parquet"
