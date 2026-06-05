import requests
from bs4 import BeautifulSoup


def diagnose():
    # 2015年1月4日 中山1R (確実に存在するレース)
    url = "https://db.netkeiba.com/race/201506010101/"
    print(f"診断中...: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'euc-jp'

        print(f"ステータスコード: {response.status_code}")
        print("-" * 20)

        # タイトルを表示（ここで「アクセスが集中しています」等が出たらBAN）
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.find("title")
        if title:
            print(f"ページタイトル: {title.get_text(strip=True)}")
        else:
            print("ページタイトル: 取得できませんでした")

        print("-" * 20)

        # 中身の判定
        if "該当するデータがありません" in response.text:
            print("判定結果: ❌ 「データなし」と返されました。")
        elif "レース結果" in response.text:
            print("判定結果: ✅ 正常です！データは見えています。")
        else:
            print("判定結果: ⚠️ 予期せぬページです（アクセス制限の可能性があります）。")
            # ページの一部を表示してみる
            print("ページの先頭100文字:", response.text[:100])

    except Exception as e:
        print(f"通信エラー: {e}")


if __name__ == "__main__":
    diagnose()