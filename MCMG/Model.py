import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from Sophia.sophia import SophiaG

from typing import Tuple, Optional

from MCMG.Configuration import ModelConfig


class SelfAttention(nn.Module):
    """Self-attention block for the GPT architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert (
            config.n_embed % config.n_head == 0
        ), f"n_embed={config.n_embed} should be a multiple of n_head={config.n_head}"

        self.config = config
        # Defining the query, key, and value linear transformations
        self.query = nn.Linear(config.n_embed, config.n_embed, bias=config.att_bias)
        self.key = nn.Linear(config.n_embed, config.n_embed, bias=config.att_bias)
        self.value = nn.Linear(config.n_embed, config.n_embed, bias=config.att_bias)

        # Dropout layers
        self.attn_drop = nn.Dropout(config.att_drop_rate)
        self.resid_drop = nn.Dropout(config.att_drop_rate)

        # Projection layer
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head

        # Registering a lower-triangular mask for causal attention
        self.mask: torch.Tensor
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, C // self.n_head)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head)

        # Flash-based implementation for scaled dot-product attention
        if self.config.do_flash:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = torch.nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.config.att_drop_rate if self.training else 0,
                is_causal=True,
            )
            y = y.transpose(1, 2)
        else:
            # Standard self-attention implementation
            # (B h T s) @ (B h s T) -> (B h T T)
            att = torch.einsum("bths,bihs->bhti", q, k) / np.sqrt(k.size(-1))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            # (B h T T) @ (B h T s) -> (B h T s)
            y = torch.einsum("bhtq,bqhs->bths", att, v)
            self.att_weights = att

        self.attended = y
        y = y.contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        self.out = y
        return y


class TransformerBlock(nn.Module):
    """GPT block that consists of one self-attention module and one MLP."""

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = SelfAttention(config)
        # Feed-forward neural network
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, config.ff_mult * config.n_embed),
            config.att_activation,
            nn.Linear(config.ff_mult * config.n_embed, config.n_embed),
            nn.Dropout(config.att_drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Main GPT model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # token, type, and position embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed)
        self.type_emb = nn.Embedding(2, config.n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embed))
        self.drop = nn.Dropout(config.gpt_drop_rate)

        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        # Final layer normalization and the output head
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=config.gpt_bias)
        self.block_size = config.block_size
        # Initializing weights
        self.apply(self._init_weights)

    def get_block_size(self):
        """Returns the block size."""
        return self.block_size

    def _init_weights(self, module):
        """Initialize weights of the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(
        self, weight_decay: float, lr: float, betas: Tuple[float, float], rho: float
    ):
        """Configure optimizers based on training configuration."""

        decay, no_decay = set(), set()
        whitelist_weight_modules = torch.nn.Linear
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Separating parameters that decay and those that don't
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias") or ("bias" in pn):
                    no_decay.add(fpn)
                elif (pn.endswith("weight") or ("weight" in pn)) and isinstance(
                    m, whitelist_weight_modules
                ):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        no_decay.add("pos_emb")

        # Checking consistency in parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert len(decay & no_decay) == 0
        assert len(param_dict.keys() - (decay | no_decay)) == 0

        # Creating optimizer groups and initializing the SophiaG optimizer
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = SophiaG(
            optim_groups, lr=lr, betas=betas, rho=rho, weight_decay=weight_decay
        )
        return optimizer

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for the GPT model."""
        b, t = idx.size()
        assert t <= self.block_size

        # Token, position, and type embeddings
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        type_embeddings = self.type_emb(
            torch.ones((b, t), dtype=torch.long, device=idx.device)
        )
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)

        # Passing through the GPT blocks
        for layer in self.blocks:
            x = layer(x)

        x = self.ln_f(x)
        logits: torch.Tensor = self.head(x)

        # Compute loss if targets are provided
        loss = (
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.loss_ignore_index,
            )
            if targets is not None
            else None
        )
        return logits, loss


    def likelihood(self, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-probabilities (log_probs) of a target sequence.

        Args:
            target (torch.Tensor): Tensor of shape (batch_size, seq_length),
                                   where each value is the token index.

        Returns:
            log_probs (torch.Tensor): Tensor of shape (batch_size,) containing
                                      the log-likelihood of each sequence in the batch.
        """
        # 获取批量大小和序列长度
        batch_size, seq_length = target.size()

        # 将目标序列分为输入和输出部分
        # 输入部分 `x` 是目标序列的前 n-1 个 token，预用于生成下一个 token
        x = target[:, :-1]  # 形状: (batch_size, seq_length - 1)
        # 输出部分 `y` 是目标序列的第 1 到 n 个 token，作为预测目标
        y = target[:, 1:]  # 形状: (batch_size, seq_length - 1)

        # 使用模型的 forward 方法获取 logits（预测分布的未归一化得分）
        logits, _ = self.forward(x)  # logits 形状: (batch_size, seq_length - 1, vocab_size)

        # 计算每个时间步的对数概率分布
        log_probs_per_step = F.log_softmax(logits, dim=-1)  # 形状: (batch_size, seq_length - 1, vocab_size)

        # 按目标序列 `y` 的索引提取其对应的对数概率
        # 注意: `y.unsqueeze(-1)` 将目标扩展为 (batch_size, seq_length - 1, 1)，以便与 log_probs_per_step 匹配
        log_probs_target = log_probs_per_step.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        # 形状: (batch_size, seq_length - 1)

        # 对序列中每个 token 的 log_probs 求和，得到每个序列的总 log-probs
        log_probs = log_probs_target.sum(dim=-1)  # 形状: (batch_size,)

        return log_probs
