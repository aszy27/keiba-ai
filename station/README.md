# keiba 発走前オッズ記録機（研究室PC）

このPC（以下「このPC」＝研究室の記録機）の役割は **`python -m scrape snapshot`（発走前オッズの記録）だけ**。
scrape weekly・odds-exotic は いつものPCのまま。

公開リポジトリなので、PC名・ユーザー名・Syncthing の機器IDは書かない。パスは次の置き換えで読む（それぞれのPCの実際のパス）:
- `<repo>`: そのPCのリポジトリ（github.com/aszy27/keiba-ai の作業ツリー）
- `<python>`: そのPCの `anaconda3\envs\keiba-ai\python.exe`
- `<keiba-station>`: このPCのユーザーフォルダの `keiba-station\`（リポジトリの外。`station_update.ps1` など、リポジトリに入れないもの）
- `<keiba-trial>`: このPCのユーザーフォルダの `keiba-trial\`（試運転の保存先）

- 本番の起動: `<repo>\station\station_run.ps1`（リポジトリの `live/run_snapshot.ps1` と同じ処理。パスだけこのPC用。ログは `logs\station\`）
- コードの更新: タスク `keiba-station-update`（毎日 3:00、`station_update.ps1`）が GitHub から自動で pull する → 下の「自動更新」
- Syncthing: `%LOCALAPPDATA%\Programs\Syncthing\syncthing.exe` v2.1.7、設定 `%LOCALAPPDATA%\Syncthing`
  - 機器IDは Syncthing の画面（操作 > ID を表示）で見る。
  - フォルダ `keiba-odds-snapshots` = `<repo>\data\v2\odds_snapshots`（送信のみ・**一時停止**・競合コピー0）。相手の機器は未登録。

## snapshot が読み書きするもの
| ファイル | 読む | 書く | 同期 |
|---|---|---|---|
| `data/v2/odds_snapshots/YYYYMMDD.csv`（その日の分だけ） | ○ 取得済みを飛ばす | ○ | **する** |
| `data/v2/odds_snapshots/exotic/YYYYMMDD_{trio,trifecta}.csv`（3連複・3連単。その日の分だけ） | ○ 取得済みを飛ばす | ○ | **する** |
| `data/v2/odds_snapshots/exotic/YYYYMMDD_{quinella,exacta}.csv`（馬連・馬単。c64e32d から。その日の分だけ） | ○ 取得済みを飛ばす | ○ | **する** |
| `data/v2/odds_snapshots/limit_state.json` | ○ | ○ | しない（PCごとの状態） |
| `data/scrape_cooldown.json` | ○ | ブロックの兆候が続いたとき | しない |
| `logs/station/snapshot_YYYYMMDD.log` | | ○ | しない |

過去の日付のファイル・race_data・parquet・models は読まない → **このPCに data のコピーは不要**。
同期の対象は `odds_snapshots.stignore`（`YYYYMMDD.csv` と `exotic\YYYYMMDD_{trio,trifecta,quinella,exacta}.csv` だけ。`*.sync-conflict-*` も除外）で決める。
`check_snapshots.ps1` が照合するのも同じファイル。`.stignore` に `!/exotic` の行は足さない（Syncthing が `!/exotic/**` に広げ、exotic\ の中が全部同期される）。
**このファイルを両方のPCの `data\v2\odds_snapshots\.stignore` に置く**（Syncthing は .stignore を同期しない）。

## 試運転（11/7〜8）
土曜 8:00 より前に起動（8:00 まで待って開始、日曜の最終レースまで動く。保存先は `<keiba-trial>\`）:
```
powershell -NoProfile -ExecutionPolicy Bypass -File <keiba-station>\trial_weekend.ps1
```
週明けに いつものPCの `20261107.csv`・`20261108.csv`（と `exotic\` の同じ日付の trio・trifecta）とレース数・status・オッズを突き合わせる。
**試運転のファイルは本番のフォルダ（`data\v2\odds_snapshots`）に絶対に入れない。**

---

## 切り替えの日（Syncthing で2台をつなぐのはこの日）

### 同じ日付のファイルが両方にあるとどうなるか
共通の履歴が無い同名ファイルの中身が違うと、Syncthing はどちらかを採り、もう片方を `*.sync-conflict-*.csv` として残すか、
受信のみ側で「ローカルの変更あり」として同期が止まる。`predict.py` はフォルダの `*.csv` を全部読むので、
**競合コピーができると判定に同じレースが2重に入る。** また、いつものPCにだけある過去の日付（10/03〜）は、受信のみ側では
「ローカルで追加されたファイル」と表示され、**「ローカルの変更を元に戻す」を押すと消える**（前向き検証のデータ）。
→ つなぐ前に「同じ日付は両方で完全に同じ」「いつものPCにだけある日付は無い」状態にする。

### 先にやること: 3連複・3連単の記録をこのPCのコードにも入れる
2026-10-26 に いつものPCで snapshot が3連複・3連単も記録するように変えた
（`data/v2/odds_snapshots/exotic/YYYYMMDD_{trio,trifecta}.csv`）。→ 0-1〜0-3 で このPCのコード・`.stignore`・`check_snapshots.ps1` も exotic/ に対応済み（2026-11-02）。
試運転（11/7〜8）でも `keiba-trial\odds_snapshots\exotic\` に3連複・3連単が書かれる。

**0-1〜0-4 が終わるまで、いつものPCの `keiba-odds-snapshot` は止めない**（止めると3連複・3連単の記録が途切れる）。

0-1. **いつものPC**: snapshot の3連複・3連単の変更をコミットして GitHub にプッシュする。
0-2. **済み（2026-11-02）** — `git pull`（6b24695）、`pytest scrape/tests` 26 passed。
   開催日でないため `--now` の実取得は未実施 → 試運転（11/7〜8）で exotic\ ができるかを見る。
   元の手順: **このPC**: `git pull` して、`--now --out-dir` で exotic/ も書かれることを確かめる（本番のフォルダには書かない）:
   ```
   cd <repo>
   & "C:\Program Files\Git\cmd\git.exe" pull
   <python> -m scrape snapshot --now --out-dir <keiba-trial>\now_check
   ```
   `now_check\YYYYMMDD.csv` に加えて `now_check\exotic\YYYYMMDD_trio.csv`・`YYYYMMDD_trifecta.csv` ができればよい。
   （開催日でないとレースが無く何も書かれないので、開催日に行うか、プッシュした変更の使い方に合わせて `--date` を付ける。）
   テストも流す: `<python> -m pytest scrape/tests -q`
0-3. **済み（2026-11-02）** — ダミーのフォルダで確かめた:
   - `check_snapshots.ps1`: exotic\ のファイルも SAME / DIFFERENT / MAIN_ONLY / STATION_ONLY が正しく出る。対象外のファイル（limit_state.json・競合コピー・3連複/3連単以外・サブフォルダ）は「not synced」に出る。
   - Syncthing（v2.1.7 の使い捨てのインスタンス2つ、送信のみ → 受信のみ）: 届いたのは `YYYYMMDD.csv` と `exotic\YYYYMMDD_{trio,trifecta}.csv` だけ。受信側の無視されるファイルは「ローカルの変更」にならない。
   - 本番のフォルダ（`data\v2\odds_snapshots\.stignore`）も置き直した（フォルダは一時停止のまま）。**いつものPCには切り替えの日の 5 で同じものを置く。**
   元の手順: **このPC**: `odds_snapshots.stignore` と `check_snapshots.ps1` に exotic/ のファイルを加える
   （`.stignore` は両方のPCに同じものを置き直す。照合は `exotic\YYYYMMDD_{trio,trifecta}.csv` も日付ごとに同一か見る）。
   照合のときに いつものPCから持ってくるファイル（下の 2・3）にも `exotic\` を含める。
0-4. 0-3 の版の `check_snapshots.ps1` を、下の手順で使う。

### 手順（0-1〜0-4 のあと。開催の無い平日に行う。月曜開催・祝日開催の日は避ける）
1. **いつものPC**: タスク `keiba-odds-snapshot` を**無効**にする（以後この日付から先は このPCだけが書く）。
2. **いつものPC**: 一覧を作る（`check_snapshots.ps1` をUSB等で持っていく）
   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File check_snapshots.ps1 -Inventory <repo>\data\v2\odds_snapshots -Out main_snapshots.csv
   ```
   できた `main_snapshots.csv` と、`odds_snapshots` の `YYYYMMDD.csv` 全部と **`exotic\` フォルダの中身全部**をこのPCへ持ってくる。
   ついでにパッケージの版も取ってくる（このPCの環境と突き合わせるため。2026-11-02 時点で未確認）:
   ```
   <python> --version > main_freeze.txt
   <python> -m pip list --format=freeze >> main_freeze.txt
   ```
   試運転の結果（`<keiba-trial>\odds_snapshots\20261107.csv`・`20261108.csv`）も、
   いつものPCの同じ日付のファイルとレース数・status・オッズを突き合わせる。
3. **このPC**: 持ってきた `YYYYMMDD.csv` を `<repo>\data\v2\odds_snapshots\` に、
   `exotic\` の中身を `...\odds_snapshots\exotic\` に置く（`limit_state.json` は持ってこない）。
4. **このPC**: 照合する
   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File <repo>\station\check_snapshots.ps1 -Compare main_snapshots.csv
   ```
   - `READY` が出るまで先へ進まない。
   - `MAIN_ONLY` → 3 のコピー漏れ。持ってきて置き直す。
   - `DIFFERENT` → **いつものPCの版を正**とする（これまで判定用に貯めてきた側）。このPCの版は別の場所へ退避してから、いつものPCの版で上書き。
   - `STATION_ONLY` → このPCが本番で取った日。問題なし（いつものPCへ送られる）。
5. **いつものPC**: Syncthing を入れ、`data\v2\odds_snapshots\.stignore` に `odds_snapshots.stignore` と同じ内容を**先に**置く。
6. 両方で相手の機器IDを追加 → このPCで `keiba-odds-snapshots` を いつものPCと共有 → いつものPCで受け入れるときに
   - フォルダの種類: **受信のみ**
   - パス: `<repo>\data\v2\odds_snapshots`
   - 詳細: 「競合ファイルの最大数」0、ファイルのバージョン管理「簡易」（置き換えられた版は `.stversions\` に残る。predict は下のフォルダを読まないので安全）
7. **このPC**: フォルダの一時停止を解除。
8. 確認（いつものPC）: フォルダが「最新の状態」、「ローカルの変更」が0件、競合コピーが無い:
   ```
   Get-ChildItem <repo>\data\v2\odds_snapshots -Recurse -Filter *sync-conflict*
   ```
   何か出たら、押す前に中身を確かめる（特に「元に戻す」は押さない）。
9. **このPC**: 下の「管理者の手順」C のタスクを有効にする。次の開催日の夜に、いつものPCに当日の CSV が届いているか見る。

以後、いつものPCでは `python -m scrape snapshot` を回さない（回すと受信のみ側のローカルの変更になり、その日の同期が止まる）。

---

## 管理者の手順（自分で行う）

### A. 休止状態を無効にする
スリープ・休止の時間は「なし」に設定済み（このセッションで実施）。休止状態そのもの（と高速スタートアップ）を切るには、
管理者の PowerShell で:
```
powercfg /hibernate off
powercfg /a
```
「休止状態は有効になっていません」と出ればよい。設定 > システム > 電源 で「電源接続時にスリープ」が「なし」であることも確認。

### B. Windows Update の再起動を開催中に起こさない
設定 > Windows Update > 詳細オプション >
- 「アクティブ時間」を **手動・6:00〜0:00**（最大18時間）。自動の再起動は 0〜6 時に寄る。
- 「更新を終了するためにできるだけ早く再起動する」を**オフ**。

再起動が起きても、C のタスクが「スタートアップ時」に、ログオンなしで再開する。

### C. タスクスケジューラ（ログオンしていなくても実行）
`taskschd.msc` → 「タスクの作成」（「基本タスク」ではない方）。

**C-1. `keiba-station-snapshot`**
- 全般: 「**ユーザーがログオンしているかどうかにかかわらず実行する**」＋「**パスワードを保存しない**」にチェック。構成: Windows 10。
  （「パスワードを保存しない」＝ S4U。インターネットには出られ、ネットワーク共有には入れない。今回は不要なので問題なし。）
- トリガー1: 毎日 8:00、「繰り返し間隔」30分・「継続時間」10時間。
- トリガー2: スタートアップ時、「遅延時間」3分（再起動・停電のあとの再開用）。
- 操作: プログラム `powershell.exe`、引数
  `-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<repo>\station\station_run.ps1"`
- 条件: 「コンピューターを AC 電源で使用している場合のみ」のチェックを外す。
- 設定: 「スケジュールされた時刻にタスクを開始できなかった場合、すぐにタスクを実行する」オン／
  「タスクを停止するまでの時間」12時間／「新しいインスタンスを開始しない」。
- 切り替えの日までは作成後に右クリック →「無効」にしておく（試運転は trial_weekend.ps1 で手動）。

**C-2. `keiba-syncthing`**（Syncthing を起動時に立ち上げる）
- 全般: C-1 と同じ（ログオンにかかわらず実行・パスワードを保存しない）。
- トリガー: スタートアップ時。
- 操作: プログラム `%LOCALAPPDATA%\Programs\Syncthing\syncthing.exe`、引数
  `serve --no-browser --no-restart --home="%LOCALAPPDATA%\Syncthing"`
- 設定: 「タスクを停止するまでの時間」のチェックを外す／「タスクが失敗した場合の再起動の間隔」1分・試行回数 最大。
- フォルダは一時停止のままなので、作ってもつなぐ日までは何も送らない。

動作確認: タスクを右クリック →「実行」→ `<repo>\logs\station\snapshot_YYYYMMDD.log` ができる
（開催日でなければ「レースが見つかりません」ですぐ終わる）。サインアウトした状態で再起動 → しばらくして
ログオンし、`Get-Process syncthing` と ブラウザの `http://127.0.0.1:8384` で Syncthing が動いていることを見る。

### D. 停電のあとに自動で起動する（BIOS）
Dell Precision 3460: 起動時に **F2** → BIOS Setup →「Power Management」→ **「AC Recovery」**（「AC Behavior」の表記もある）を
**「Power On」**（または「Last Power State」）→ Apply → Exit。
電源が戻るとPCが起動し、C のタスク（スタートアップ時）がログオンなしで動く。

### E. BitLocker の確認
起動時に PIN を求める設定だと、停電後の自動起動がそこで止まる。管理者の PowerShell で `manage-bde -status C:` を見て、
「キーの保護機能」が TPM だけなら問題なし（TPM と PIN なら PIN を外すかどうか判断する）。

---

## まとめて設定するスクリプト（2026-11-02 追加）
管理者の手順の A・C・E と試運転のタスクは `setup_admin.ps1` で一度に行える（B・D は手で）。管理者の PowerShell で:
```
powershell -NoProfile -ExecutionPolicy Bypass -File <keiba-station>\setup_admin.ps1
```
- `keiba-station-snapshot`（本番）は**無効**で作り、1回だけ動かして確かめてからまた無効に戻す。
- `keiba-station-trial`: 11/7・11/8 の 7:30 とスタートアップ時に `trial_weekend.ps1 -Dates 20261107,20261108` を動かす（保存先は keiba-trial。再起動しても続きから）。**このタスクがあるので trial_weekend.ps1 を手で起動しない**（2重に動く）。試運転が終わったら削除する。

---

## 自動更新（keiba-station-update、2026-11-02 追加）
タスク `keiba-station-update`（毎日 3:00、ログオンしていなくても実行・パスワードを保存しない）が
`<keiba-station>\station_update.ps1` を動かす。作るのは管理者の PowerShell で:
```
powershell -NoProfile -ExecutionPolicy Bypass -File <keiba-station>\setup_update_task.ps1
```
（作ったあと1回だけ実行して結果を表示する。これは本当の更新＝ git pull とテスト。）

### 何をするか
1. 次のときは何もしない（SKIP）:
   - snapshot・試運転が動いている（`scrape snapshot`・`station_run.ps1`・`trial_weekend.ps1`・`run_snapshot.ps1` のプロセスがある）
   - 8:00〜18:00 で、その日が開催日（3:00 に電源が切れていて遅れて動いたとき。netkeiba のレース一覧で判定し、判定できなければ開催日とみなす）
2. 追跡しているファイルにこのPCでの変更があれば pull しない（ERROR）。
3. `git pull --ff-only`（リポジトリは公開なので認証なし。資格情報の入力画面は出さない設定で呼ぶ）。
4. `python -m pytest scrape/tests -q`
5. 通ったら `<repo>\station\odds_snapshots.stignore` を本番の `data\v2\odds_snapshots\.stignore` にコピー（中身が変わったときだけ。
   最後の行が `*`・`*.sync-conflict-*` を含む、を満たさなければコピーしない）。`station\` がまだリポジトリに無いうちは飛ばす。
6. テストが落ちたら `git reset --hard <pull の前のコミット>` で戻す（ROLLBACK）。`.stignore` は触らない。
   戻したあとも次の日の 3:00 にまた pull してテストする → いつものPCで直してプッシュすれば、翌日に自動で取り込まれる。

`station_update.ps1` 自体は**リポジトリに入れない**（壊れたコミットが「戻す役」を置き換えられないように）。

### ログ（すべて `<repo>\logs\station\`。Git 管理外・同期しない）
| ファイル | 中身 |
|---|---|
| `update_history.log` | 1回の実行につき1行。`OK` / `SKIP` / `ROLLBACK` / `ERROR` と、pull の前後のコミット |
| `update_YYYYMMDD.log` | その日の実行の全部（pull の出力・取り込んだコミット・pytest の出力） |
| `stignore_backup\.stignore_YYYYMMDD_HHMMSS` | `.stignore` を置き換える前の版 |
| `pytest_tmp\` | テストの一時フォルダ（毎回消して作り直す） |
| `snapshot_YYYYMMDD.log` | （参考）本番の snapshot のログ |

まず見るのは:
```
Get-Content <repo>\logs\station\update_history.log -Tail 10 -Encoding UTF8
```

### 失敗したとき
**ROLLBACK（テストが落ちた）** — このPCは pull の前のコミットに戻っていて、snapshot はそのコードで動く。急ぎの作業は無い。
1. `update_YYYYMMDD.log` の pytest の出力で、どのテストが落ちたかを見る。
2. いつものPCで直してプッシュ → 翌日 3:00 に自動で取り込まれる。すぐ取り込むなら、開催日の 8:00〜18:00 以外に:
   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File <keiba-station>\station_update.ps1
   ```
   （タスクから動かすなら `Start-ScheduledTask keiba-station-update`。）
3. このPCだけで落ちる場合（パッケージの版の違いなど）は、いつものPCの `main_freeze.txt` と `pip list` を突き合わせる。

**ERROR**
- `tracked files modified locally` → このPCでリポジトリのファイルを直接変えている。`git -C <repo> status` で確かめ、
  要らない変更なら `git -C ... checkout -- <ファイル>`（必要な変更ならいつものPCに持っていってコミットする）。
- `git pull failed` → ネットワークか、このPCにだけコミットがある（`--ff-only` は合流しない）。HEAD は変わっていない。
  `git -C ... log --oneline origin/main..HEAD` で、このPCにだけあるコミットを確かめる。
- `rollback failed` → 手で戻す（下）。
- `.stignore looks wrong` → テストは通ったが、リポジトリの `station\odds_snapshots.stignore` がおかしい。コードは新しい版のまま、
  本番の `.stignore` は前のまま。いつものPCで直してプッシュする。

**手で戻す**（自動の戻しが失敗したとき・取り込んだあとで問題に気づいたとき）:
```
cd <repo>
& "C:\Program Files\Git\cmd\git.exe" log --oneline -5          # update_history.log の「back to」/「updated A..B」の A が戻す先
& "C:\Program Files\Git\cmd\git.exe" reset --hard <戻す先のコミット>
<python> -m pytest scrape/tests -q
```
`reset --hard` は追跡しているファイルだけを戻す。`data\`・`logs\`（Git 管理外）は消えない。
戻しても翌日 3:00 にまた pull されるので、直るまで取り込みたくなければ `Disable-ScheduledTask keiba-station-update`（管理者）で止める。
`.stignore` を前の版に戻すなら `logs\station\stignore_backup\` の最新のものを `data\v2\odds_snapshots\.stignore` にコピーする。

---

## リポジトリの `station\` へ移した（2026-11-02、いつものPCでコミット）
このPC専用のファイルを `<repo>\station\` に入れて、GitHub 経由で（上の自動更新で）届くようにした。

### 移したもの
| 元（`<keiba-station>\`） | 移した先（`<repo>\station\`） | 移すときに変えたこと |
|---|---|---|
| `odds_snapshots.stignore` | `station\odds_snapshots.stignore` | 馬連・馬単の2行（下） |
| `check_snapshots.ps1` | `station\check_snapshots.ps1` | 馬連・馬単（下）。`-StationDir` の既定値はこのPCのパスのまま |
| `station_run.ps1` | `station\station_run.ps1` | `$proj` を `Split-Path $PSScriptRoot -Parent` に（置き場所から求める）。`$py` はこのPCのパスのまま |
| `README.md` | `station\README.md` | パスを `station\` と置き換えの書き方（`<repo>` など）に。PC名・ユーザー名・機器IDを消した |

**移さないもの**（`<keiba-station>\` に残す）: `station_update.ps1`（戻す役なのでリポジトリの外）、`setup_admin.ps1`・`setup_update_task.ps1`
（1回きり）、`trial_weekend.ps1`（試運転が終わったら消す）。README の代わりに「本体は `<repo>\station\README.md`」の1行だけのメモを置く。

### 一緒にコミットしたもの
- `.gitattributes` に `*.stignore text eol=lf`（Windows の checkout でも LF のまま。0-3 で Syncthing で確かめたファイルとバイト単位で同じにする）。
- **馬連・馬単**: c64e32d（2026-11-02）で snapshot が `exotic\YYYYMMDD_{quinella,exacta}.csv` も書くようになったので、
  `.stignore` に `!/exotic/[0-9]...[0-9]_quinella.csv`・`_exacta.csv` の2行（trio・trifecta と同じ書き方。`!/exotic` は足さない）、
  `check_snapshots.ps1` の `$SyncedRe` を `(trio|trifecta|quinella|exacta)` にした。馬連・馬単のファイルがあるのは 2026-11-07 から。
  **いつものPCには Syncthing が無いので、使い捨ての Syncthing での確認はしていない** → 下の手順 2 でこのPCが行う。

### 残りの手順
1. ~~いつものPC: 上の4つを `station\` に置き、`.gitattributes` の行を足してコミット・プッシュ。~~ 済み（2026-11-02）
2. **このPC**: 翌日 3:00 の自動更新で届く（すぐなら `station_update.ps1` を手で動かす）。`update_history.log` が
   `stignore: UPDATED` になることを見る（フォルダは一時停止のままなので、`.stignore` が置き換わっても何も送られない）。
   0-3 と同じく使い捨ての Syncthing で、`exotic\YYYYMMDD_{quinella,exacta}.csv` が届き、それ以外の exotic\ のファイルは届かないことを確かめる。
   `check_snapshots.ps1` もダミーのフォルダで、quinella・exacta が SAME / DIFFERENT / MAIN_ONLY / STATION_ONLY に出ることを確かめる。
3. **このPC**（管理者）: `keiba-station-snapshot` の操作の引数を
   `-File "<repo>\station\station_run.ps1"` に変える（`setup_admin.ps1` の `$station` も合わせる）。
4. 1回動かしてログを確かめたら、`<keiba-station>\` の4つの古いファイルを消す（2重に持たない）。
