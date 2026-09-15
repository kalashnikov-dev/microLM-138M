import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchtune.modules import RotaryPositionalEmbeddings


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int = 2048
    d_model: int = 768
    d_head: int = 64
    n_heads: int = 12 
    n_kv_heads: int = 4
    n_layers: int = 16
    eps_norm: float = 1e-5
    std: float = 0.02

    @property
    def d_ff(self) -> int:
        return int(self.d_model * 8 / 3) # 2048

    @property 
    def scaled_std(self) -> float:
        return self.std / math.sqrt(2 * self.n_layers) # for residual projections 


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.atn_norm = nn.RMSNorm(config.d_model, eps=config.eps_norm)  # https://docs.pytorch.org/docs/2.13/generated/torch.nn.RMSNorm.html, eps for bfloat stability
        self.q = nn.Linear(config.d_model, config.d_head * config.n_heads, bias=False)
        self.k = nn.Linear(config.d_model, config.d_head * config.n_kv_heads, bias=False)
        self.v = nn.Linear(config.d_model, config.d_head * config.n_kv_heads, bias=False)
        self.o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.q_norm = nn.RMSNorm(config.d_head, eps=config.eps_norm)
        self.k_norm = nn.RMSNorm(config.d_head, eps=config.eps_norm) 
        self.pre_ff_norm = nn.RMSNorm(config.d_model, eps=config.eps_norm)

        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)

        nn.init.normal_(self.q.weight, mean=0.0, std=config.std)
        nn.init.normal_(self.k.weight, mean=0.0, std=config.std)
        nn.init.normal_(self.v.weight, mean=0.0, std=config.std)
        nn.init.normal_(self.w1.weight, mean=0.0, std=config.std)
        nn.init.normal_(self.w3.weight, mean=0.0, std=config.std)

        nn.init.normal_(self.o.weight, mean=0.0, std=config.scaled_std)
        nn.init.normal_(self.w2.weight, mean=0.0, std=config.scaled_std)


    def forward(self, x, rope):
        b, s, _ = x.shape

        x_norm = self.atn_norm(x)

        Q = self.q_norm(self.q(x_norm).view(b, s, self.config.n_heads, self.config.d_head))
        K = self.k_norm(self.k(x_norm).view(b, s, self.config.n_kv_heads, self.config.d_head))
        V = self.v(x_norm).view(b, s, self.config.n_kv_heads, self.config.d_head)

        Q_rotated = rope(Q)
        K_rotated = rope(K)

        x_atn = F.scaled_dot_product_attention(
            Q_rotated.transpose(1, 2), K_rotated.transpose(1, 2), V.transpose(1, 2), is_causal=True, enable_gqa=True)  # https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html

        x_atn = self.o(x_atn.transpose(1, 2).reshape(b, s, self.config.d_head * self.config.n_heads))

        x = x + x_atn

        x_norm = self.pre_ff_norm(x)

        swiglu = self.w2(self.w1(x_norm) * F.silu(self.w3(x_norm)))

        x = x + swiglu
        return x



class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.lm_head = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryPositionalEmbeddings(dim=config.d_head, max_seq_len=config.seq_len + 1)  # https://meta-pytorch.org/torchtune/stable/generated/torchtune.modules.RotaryPositionalEmbeddings.html?highlight=rope
        self.post_ff_norm = nn.RMSNorm(config.d_model, eps=config.eps_norm)
        self.ff = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.std)


    def forward(self, tokens):
        x = self.lm_head(tokens)

        for block in self.ff:
            #x = block(x, self.rope)
            x = checkpoint(block, x, self.rope, use_reentrant=False)  # 15% tok/s gain
        x = self.post_ff_norm(x)

        return x