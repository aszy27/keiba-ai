import pandas as pd

from v2 import ingest


def _raw():
    return pd.DataFrame({
        "race_id": ["202601010101", "202601010101", "202601010102", "202601010103", "202601010104"],
        "horse_id": ["2020100001", "2020100002", "2020100003", "2020100004", "2020100007"],
        "type": [None, None, "芝", "ダート", None],
        "condition": [None, None, "良", "重", None],
        "race_name": ["3КаЬЄОЁЭј", "3КаЬЄОЁЭј", "3歳未勝利", "3歳以上1勝クラス", "TVhÇŐ(3ŸĄ)"],
    })


def test_apply_patches(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PATCH_DIR", tmp_path)
    pd.DataFrame({"race_id": ["202601010103"], "reason": ["偽レース"]}).to_csv(tmp_path / "race_delete.csv", index=False)
    (tmp_path / "race_rows").mkdir()
    pd.DataFrame({"race_id": ["202601010102"] * 2, "horse_id": ["2020100005", "2020100006"], "type": ["芝"] * 2,
                  "condition": ["稍"] * 2, "race_name": ["3歳未勝利"] * 2}).to_csv(tmp_path / "race_rows" / "202601010102.csv", index=False)
    pd.DataFrame({"race_id": ["202601010101", "202601010104"], "type": ["ダート", "芝"], "condition": ["", "良"],
                  "race_name": ["3歳未勝利", "TVh杯"]}).to_csv(tmp_path / "race_info.csv", index=False)

    notes = []
    out = ingest.apply_patches(_raw().astype("object"), notes)

    assert "202601010103" not in set(out["race_id"])
    assert sorted(out.loc[out["race_id"] == "202601010102", "horse_id"]) == ["2020100005", "2020100006"]
    first = out[out["race_id"] == "202601010101"]
    assert (first["type"] == "ダート").all()
    assert first["condition"].isna().all()                  # パッチの空欄は元の値を上書きしない
    assert (first["race_name"] == "3КаЬЄОЁЭј").all()       # 戻せる文字化けは置き換えない（クラス表記が残るため）
    fourth = out[out["race_id"] == "202601010104"]
    assert fourth["race_name"].tolist() == ["TVh杯"]          # 戻せない文字化けは置き換える
    assert len(notes) == 3


def test_apply_patches_without_patch_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "PATCH_DIR", tmp_path / "none")
    raw = _raw()
    assert ingest.apply_patches(raw, []).equals(raw)
