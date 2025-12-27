import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
import math


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, 2*hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, in_dim, bias=False)
    def forward(self, x):
        x = self.in_proj(x)
        gate, up = x.chunk(2, dim=-1)
        return self.out_proj(up * F.silu(gate))


class GQA(nn.Module):
    def __init__(
        self,
        q_embed_dim,
        kv_embed_dim,
        out_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        compute_dtype,
    ):
        super().__init__()
        self.q_embed_dim = q_embed_dim
        self.kv_embed_dim = kv_embed_dim
        self.out_embed_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_gqa_groups = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(q_embed_dim, num_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(kv_embed_dim, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_query_heads * head_dim, out_embed_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        orig_dtype = x.dtype

        q = self.q_proj(x).view(batch_size, seq_len, self.num_query_heads, self.head_dim)   # (B, T, N, H)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # q = self.rotary_emb(q, q_pos)
        # k = self.rotary_emb(k, kv_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            #enable_gqa=self.num_gqa_groups > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_embed_dim)
        output = self.out_proj(attn_output)
        return output.to(orig_dtype)


class GQCA(nn.Module):
    def __init__(
        self,
        q_embed_dim,
        kv_embed_dim,
        out_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        compute_dtype,
    ):
        super().__init__()
        self.q_embed_dim = q_embed_dim
        self.kv_embed_dim = kv_embed_dim
        self.out_embed_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_gqa_groups = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(q_embed_dim, num_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(kv_embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(kv_embed_dim, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_query_heads * head_dim, out_embed_dim, bias=False)

    def forward(self, xq, xkv):
        batch_size, seq_len_q, _ = xq.shape
        orig_dtype = xq.dtype

        _, seq_len_kv, _ = xkv.shape

        q = self.q_proj(xq).view(batch_size, seq_len_q, self.num_query_heads, self.head_dim)   # (B, T, N, H)
        k = self.k_proj(xkv).view(batch_size, seq_len_kv, self.num_kv_heads, self.head_dim)
        v = self.v_proj(xkv).view(batch_size, seq_len_kv, self.num_kv_heads, self.head_dim)

        # q = self.rotary_emb(q, q_pos)
        # k = self.rotary_emb(k, kv_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            # enable_gqa=self.num_gqa_groups > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.q_embed_dim)   # note
        output = self.out_proj(attn_output)
        return output.to(orig_dtype)


class MultiheadAttentionPoolingLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        enc_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        ca_num_query_heads,
        ca_num_kv_heads,
        ca_head_dim,
        expand_dim,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps

        self.pre_sa_norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )
        self.sa = GQA(
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            out_embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            compute_dtype=torch.float32,
        )
        
        self.pre_ca_norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )
        self.ca = GQCA(
            q_embed_dim=embed_dim,
            kv_embed_dim=enc_embed_dim,
            out_embed_dim=embed_dim,
            num_query_heads=ca_num_query_heads,
            num_kv_heads=ca_num_kv_heads,
            head_dim=ca_head_dim,
            compute_dtype=torch.float32,
        )
        
        self.pre_mlp_norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )
        self.mlp = MLP(
            in_dim=embed_dim,
            hidden_dim=expand_dim,
        )

    def forward(self, x, patch_tokens):
        residual = x
        x_norm = self.pre_sa_norm(x)
        sa_out = self.sa(x=x_norm)
        x = residual + sa_out

        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out = self.ca(xq=x_norm, xkv=patch_tokens)
        x = residual + ca_out

        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class AttentionPooler(nn.Module):
    def __init__(self, num_queries, embed_dim, patch_embed_dim, num_heads, num_layers, norm_eps=1e-5):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.abs_pos_embed = nn.Embedding(75, embed_dim)
        self.layers = nn.ModuleList([
            MultiheadAttentionPoolingLayer(
                embed_dim=embed_dim,
                enc_embed_dim=patch_embed_dim,
                num_query_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=embed_dim // num_heads,
                ca_num_query_heads=num_heads,
                ca_num_kv_heads=num_heads,
                ca_head_dim=embed_dim // num_heads,
                expand_dim=4 * embed_dim,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(
            embed_dim,
            eps=norm_eps,
            dtype=torch.float32,
        )

    def forward(self, patch_tokens):
        B, T = patch_tokens.shape[0], patch_tokens.shape[1]
        query_emb = self.query_embed.expand(B, -1, -1)
        pos = torch.arange(75, device=patch_tokens.device)
        pos_emb = self.abs_pos_embed(pos)

        query_emb = query_emb + pos_emb.unsqueeze(0)
        for layer in self.layers:
            query_emb = layer(
                x=query_emb,
                patch_tokens=patch_tokens,
            )
        query_emb = self.norm(query_emb)

        return query_emb