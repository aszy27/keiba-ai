# train_graph_embedding.py
import os
import sys
import hashlib

# ==========================================
# プロジェクトルートの動的解決とパス追加
# ==========================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
import random
import warnings
import numpy as np
import logging

from core.config import VEC_DIM

warnings.simplefilter('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ==========================================
# 設定
# ==========================================
VECTOR_SIZE = VEC_DIM
WALK_LENGTH = 30
NUM_WALKS = 10
WINDOW_SIZE = 5
MIN_COUNT = 2
WORKERS = 1  # 再現性確保のため1に固定

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "horse_vectors.csv")
INPUT_FILE = os.path.join(DATA_DIR, "master_horse_data.csv")


def hashfxn(x):
    """
    ★ 不変ハッシュ関数
    Pythonの環境変数に依存せず、別PCや別日に実行しても100%同じベクトル空間を生成する。
    """
    return int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16)


class WalkGenerator:
    """
    ★ メモリ最適化ランダムウォーク・ジェネレータ
    Word2Vecがエポックを回すたびに、オンザフライでウォークを生成する。
    """
    def __init__(self, G, num_walks, walk_length):
        print("   -> 隣接リストを事前キャッシュ中 (Generator Init)...")
        self.nodes = sorted(list(G.nodes()))
        self.adj_list = {n: sorted(list(G[n])) for n in self.nodes}
        self.num_walks = num_walks
        self.walk_length = walk_length

    def __iter__(self):
        # バグ修正: tqdmによるログ崩壊を防ぐため、シンプルなループに変更。
        # 進捗はGensimの内部ロギング（INFO）が綺麗に出力してくれます。
        for _ in range(self.num_walks):
            random.shuffle(self.nodes)
            for node in self.nodes:
                walk = [node]
                while len(walk) < self.walk_length:
                    cur = walk[-1]
                    neighbors = self.adj_list[cur]
                    if len(neighbors) > 0:
                        walk.append(random.choice(neighbors))
                    else:
                        break
                yield [str(w) for w in walk]


def main():
    # シードの完全固定
    random.seed(42)
    np.random.seed(42)

    print("🧬 血統の数値化（グラフ埋め込み: DeepWalk）を開始します...")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ {INPUT_FILE} が見つかりません。scrape_horse_ped.py を実行してください。")
        return

    if os.path.exists(OUTPUT_FILE):
        print(f"⚠️ {OUTPUT_FILE} が既に存在します。上書きします。")
        print("   ⚠️ この処理の完了後、train_main.py の再実行が必要です。")

    try:
        # Windows環境での文字化けKeyErrorを防ぐため encoding を明示
        df = pd.read_csv(INPUT_FILE, dtype=str, encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ 読み込みエラー: {e}")
        return

    print(f"   -> {len(df)}頭の血統データを確認しました。")

    print("🕸️ 家系図ネットワークを構築中...")
    G = nx.Graph()

    INVALID_IDS = ['nan', 'unknown', '', '0', 'none']

    def get_valid_edges(source_col, target_col):
        if target_col not in df.columns or source_col not in df.columns:
            return []
        edges = df[[source_col, target_col]].copy()
        edges[source_col] = edges[source_col].astype(str).str.strip()
        edges[target_col] = edges[target_col].astype(str).str.strip()

        valid_mask = (
                ~edges[source_col].str.lower().isin(INVALID_IDS) &
                ~edges[target_col].str.lower().isin(INVALID_IDS)
        )
        valid_edges = edges[valid_mask]
        return list(zip(valid_edges[source_col], valid_edges[target_col]))

    # バグ修正: マスタに実在する1代前の親子エッジのみを抽出
    edges_sire = get_valid_edges('horse_id', 'sire_id')
    edges_dam = get_valid_edges('horse_id', 'dam_id')

    # 1代前の親子エッジを全て入れるだけで、NetworkXが無向グラフ上で
    # 自動的に祖父母、曾祖父母、それ以上の世代まで家系図を一本のネットワークに連結します。
    G.add_edges_from(edges_sire)
    G.add_edges_from(edges_dam)

    num_nodes = G.number_of_nodes()
    print(f"   -> ノード数(馬): {num_nodes}, エッジ数(親子関係): {G.number_of_edges()}")

    if num_nodes < 10:
        print("⚠️ データが少なすぎます。scrape_horse_ped.py でデータを収集してください。")
        return

    print(f"🚶 メモリ最適化済みランダムウォーク・ジェネレータを起動 (Walk={WALK_LENGTH}, N={NUM_WALKS})...")
    walk_generator = WalkGenerator(G, NUM_WALKS, WALK_LENGTH)

    print("🧠 血統ベクトル学習 (Word2Vec skip-gram)...")
    model = Word2Vec(
        sentences=walk_generator,
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1,
        hs=0,
        negative=5,
        epochs=10,
        seed=42,
        hashfxn=hashfxn
    )

    print(f"💾 '{OUTPUT_FILE}' に保存中...")
    vectors = []
    header = ['horse_id'] + [f'p_vec_{i}' for i in range(VECTOR_SIZE)]

    vocab = sorted(list(model.wv.index_to_key))
    for node in vocab:
        vec = model.wv[node]
        vectors.append([node] + vec.tolist())

    df_vec = pd.DataFrame(vectors, columns=header)
    df_vec.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"✅ 完了！ {len(df_vec)}頭の血統ベクトルを生成しました。")


if __name__ == "__main__":
    main()