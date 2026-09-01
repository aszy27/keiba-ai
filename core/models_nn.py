# core/models_nn.py
import torch
import torch.nn as nn
import math


class StrongDAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.PReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.BatchNorm1d(hidden_dim // 4), nn.PReLU(),
            nn.Linear(hidden_dim // 4, latent_dim), nn.BatchNorm1d(latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4), nn.BatchNorm1d(hidden_dim // 4), nn.PReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.PReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        z = self.encoder(x)
        return z, self.decoder(z)


class RacingTransformer(nn.Module):
    def __init__(self, cat_dims, num_dim, hist_dim, seq_len=5, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.hist_len = seq_len
        self.d_model = d_model

        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(50, (dim + 1) // 2)) for dim in cat_dims
        ])

        self.hist_input_proj = nn.Linear(hist_dim, d_model)

        # 🔴 FIX (コメント訂正): 実際の格納順は「index 末尾 = 最新」(右詰め)。
        #   build_hist_array / build_hist_tensor いずれも
        #   hist_tensor[i, hist_len-seq_length:, :] = [最古, ..., 最新] と右詰めしている。
        #   そのため index 0 はパディング or 最古、index seq_len-1 が最新。
        #   位置エンコーディングは正順 (position=0 → 最古側) のままで正しい。
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0)) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pos_encoder', pe.unsqueeze(0))

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
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]

        padding_mask = (x_hist.abs().sum(dim=-1) == 0)

        h = self.hist_input_proj(x_hist) + self.pos_encoder

        all_masked = padding_mask.all(dim=1)
        if all_masked.any():
            h = torch.where(all_masked.view(-1, 1, 1), self.debut_embedding, h)
            padding_mask = padding_mask.masked_fill(all_masked.unsqueeze(1), False)

        h = self.transformer_encoder(h, src_key_padding_mask=padding_mask)

        h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        valid_lens = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        h_pooled = h.sum(dim=1) / valid_lens

        return self.fc(torch.cat([h_pooled] + emb_list + [x_num], dim=1))