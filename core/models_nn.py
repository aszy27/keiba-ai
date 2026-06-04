# core/models_nn.py
import torch
import torch.nn as nn
import math


class StrongDAE(nn.Module):
    """
    Denoising Autoencoder (軽量・高速版)
    """

    def __init__(self, input_dim, hidden_dim=256, latent_dim=32, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.PReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.BatchNorm1d(hidden_dim // 4), nn.PReLU(),
            nn.Linear(hidden_dim // 4, latent_dim), nn.BatchNorm1d(latent_dim)  # 特徴量スケール安定化
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4), nn.BatchNorm1d(hidden_dim // 4), nn.PReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.PReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, input_dim)  # 復元目標は正規化済みのため線形出力
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        z = self.encoder(x)
        return z, self.decoder(z)


class RacingTransformer(nn.Module):
    """
    カテゴリ・数値・時系列（過去成績）を統合した競馬予測 Transformer (推論最適化版)
    """

    def __init__(self, cat_dims, num_dim, hist_dim, seq_len=5, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.hist_len = seq_len
        self.d_model = d_model

        # OOV対策: max_norm 等を使わず軽量なEmbedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(50, (dim + 1) // 2)) for dim in cat_dims
        ])

        self.hist_input_proj = nn.Linear(hist_dim, d_model)

        # 位置エンコーディング
        # 🟡 FIX: 以前は arange(seq_len-1, -1, -1) で逆順(最新に最大値)だった。
        #          build_hist_array の格納順(index 0 = 最新)に合わせて正順に修正。
        #          index 0(最新) → position=0, index seq_len-1(最古) → position=seq_len-1
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0)) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pos_encoder', pe.unsqueeze(0))

        # デビュー馬（過去データなし）専用の学習可能埋め込み
        self.debut_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=128,
                                       dropout=0.1),
            num_layers=num_layers
        )

        emb_total = sum(emb.embedding_dim for emb in self.embeddings)
        fc_input_dim = d_model + emb_total + num_dim

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_cat, x_num, x_hist):
        # 1. カテゴリ変数のEmbedding結合（推論時は既にクリップ済みを前提とし、高速化）
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]

        # 2. パディングマスク生成
        # x_hist が全て0の行を True とする
        padding_mask = (x_hist.abs().sum(dim=-1) == 0)

        # 3. 過去履歴の投影と位置エンコーディング加算
        h = self.hist_input_proj(x_hist) + self.pos_encoder

        # 4. デビュー馬（全ステップがパディング）への埋め込み適用 (cloneを避けた最適化)
        all_masked = padding_mask.all(dim=1)
        if all_masked.any():
            # boolean mask を使ってインプレースに近い形で書き換え
            h = torch.where(all_masked.view(-1, 1, 1), self.debut_embedding, h)
            # デビュー馬はパディングとして扱わないようにマスクを解除
            padding_mask = padding_mask.masked_fill(all_masked.unsqueeze(1), False)

        # 5. Transformer エンコーダー通過
        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        # 6. パディング位置の出力をゼロクリア
        h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # 7. Flatten の代わりに Global Average Pooling を使用
        # 107行目付近
        # パディングされていない要素数を計算 (0除算防止のため .clamp(min=1) を使用)
        valid_lens = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)

        # 🔴 FIX: unsqueeze(-1) を削除！
        # h.sum(dim=1) は [B, D] (2次元)、valid_lens は [B, 1] (2次元) なので、
        # そのまま割ることで 2次元 ([B, D]) を維持したまま平均化できます。
        h_pooled = h.sum(dim=1) / valid_lens

        return self.fc(torch.cat([h_pooled] + emb_list + [x_num], dim=1))