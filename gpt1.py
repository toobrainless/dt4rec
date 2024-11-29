import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    DEVICE = query.device

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(DEVICE)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(DEVICE)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_layer = 6
    n_head = 8
    n_embd = 64
    memory_size = 3

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, **kwargs):
        """
        Arguments setter
        """
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        embed_dimension: int,
        bias: bool = False,
        is_causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
            is_causal = self.is_causal
        else:
            dropout = 0.0
            is_causal = False

        y = scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attn = CausalSelfAttention(
            num_heads=config.n_head,
            embed_dimension=config.n_embd,
            dropout=config.attn_pdrop,
            is_causal=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        """
        :x: batch
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class StateReprModule(nn.Module):
    """
    Compute state for RL environment. Based on `DRR paper
    <https://arxiv.org/pdf/1810.12027.pdf>`_

    Computes State is a concatenation of user embedding,
    weighted average pooling of `memory_size` latest relevant items
    and their pairwise product.
    """

    def __init__(
        self,
        user_num,
        item_num,
        embedding_dim,
        memory_size,
    ):
        super().__init__()
        self.user_embeddings = nn.Embedding(user_num, embedding_dim)

        self.item_embeddings = nn.Embedding(
            item_num + 1, embedding_dim, padding_idx=int(item_num)
        )

        self.drr_ave = torch.nn.Conv1d(
            in_channels=memory_size, out_channels=1, kernel_size=1
        )

        self.linear = nn.Linear(3 * embedding_dim, embedding_dim)

        self.initialize()

    def initialize(self):
        """weight init"""
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        self.item_embeddings.weight.data[-1].zero_()

        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.uniform_(self.drr_ave.weight)

        self.drr_ave.bias.data.zero_()

    def forward(self, user, memory):
        """
        :param user: user batch
        :param memory: memory batch
        :return: vector of dimension embedding_dim
        """
        user_embedding = self.user_embeddings(user.long()).squeeze(1)
        item_embeddings = self.item_embeddings(memory.long())
        drr_ave = self.drr_ave(item_embeddings).squeeze(1)
        output = torch.cat((user_embedding, user_embedding * drr_ave, drr_ave), 1)
        output = self.linear(output)

        return output


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.user_num = config.user_num
        self.memory_size = config.memory_size

        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size + 1, config.n_embd)
        )
        self.global_pos_emb = nn.Parameter(
            torch.zeros(1, config.max_timestep + 1, config.n_embd)
        )
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

        self.state_repr = StateReprModule(
            user_num=config.user_num,
            item_num=config.vocab_size,
            embedding_dim=config.n_embd,
            memory_size=config.memory_size,
        )

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(
            self.state_repr.item_embeddings, nn.Tanh()
        )
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        """
        Return block_size
        """
        return self.block_size

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.Conv1d,
        )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("global_pos_emb")

        # validate that we considered every parameter
        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {inter_params!s} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {param_dict.keys() - union_params!s} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": 0.1,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    # state, action, and return
    def calc_hidden_state(
        self,
        states,
        actions,
        rtgs,
        timesteps,
        users,
    ):
        """
        :states: states batch, (batch, trajectory_len, 3)
        :actions: actions batch, (batch, trajectory_len, 1)
        :rtgs: rtgs batch, (batch, trajectory_len, 1)
        :timesteps: timesteps batch, (batch, 1, 1)
        :users:users batch,  (batch, 1)
        """

        inference = not self.training
        trajectory_len = states.shape[1]
        state_embeddings = self.state_repr(
            users.repeat((1, trajectory_len)).reshape(-1, 1),
            states.reshape(-1, 3),
        )

        state_embeddings = state_embeddings.reshape(
            states.shape[0], states.shape[1], self.config.n_embd
        )  # (batch, block_size, n_embd)

        if actions is not None:
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(
                actions.type(torch.long).squeeze(-1)
            )  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (
                    states.shape[0],
                    states.shape[1] * 3 - int(inference),
                    self.config.n_embd,
                ),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::3, :] = rtg_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings[
                :, -states.shape[1] + int(inference) :, :
            ]
        else:
            # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros(
                (states.shape[0], states.shape[1] * 2, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device,
            )
            token_embeddings[:, ::2, :] = rtg_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(
            self.global_pos_emb, batch_size, dim=0
        )  # batch_size, traj_length, n_embd

        position_embeddings = (
            torch.gather(
                all_global_pos_emb,
                1,
                torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1),
            )
            + self.pos_emb[:, : token_embeddings.shape[1], :]
        )
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        if actions is not None:
            x = x[:, 1::3, :]  # only keep predictions from state_embeddings
        elif actions is None:
            x = x[:, 1:, :]

        return x

    def forward(
        self,
        states,
        actions,
        rtgs,
        timesteps,
        users,
    ):
        x = self.calc_hidden_state(states, actions, rtgs, timesteps, users)
        logits = self.head(x)

        return logits

    def predict(self, states, actions, rtgs, timesteps, users):
        """
        :states: states batch, (batch, block_size, 3)
        :actions: actions batch, (batch, block_size, 1)
        :rtgs: rtgs batch, (batch, block_size, 1)
        :timesteps: timesteps batch, (batch, 1, 1)
        :users:users batch,  (batch, 1)
        """
        logits, _ = self(
            states=states.to(self.pos_emb.device),
            actions=actions.to(self.pos_emb.device),
            targets=None,
            rtgs=rtgs.to(self.pos_emb.device),
            timesteps=timesteps.to(self.pos_emb.device),
            users=users.to(self.pos_emb.device),
        )
        logits = logits[:, -1, :]
        actions = logits.argsort(dim=1, descending=True)
        return actions
