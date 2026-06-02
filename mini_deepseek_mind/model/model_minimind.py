import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
        ### mHC (Manifold-Constrained Hyper-Connections)
        self.use_mhc = kwargs.get("use_mhc", True)
        self.mhc_n_hc = kwargs.get("mhc_n_hc", 4)
        self.mhc_sinkhorn_iters = kwargs.get("mhc_sinkhorn_iters", 20)
        ### CSA / HCA (Compressed Sparse / Heavily Compressed Attention)
        self.use_csa_hca = kwargs.get("use_csa_hca", False)
        self.csa_m = kwargs.get("csa_m", 4)
        self.csa_top_k = kwargs.get("csa_top_k", 64)
        self.csa_d_c = kwargs.get("csa_d_c", 128)
        self.csa_n_ih = kwargs.get("csa_n_ih", 4)
        self.csa_c_I = kwargs.get("csa_c_I", 32)
        self.csa_g = kwargs.get("csa_g", 4)
        self.csa_d_g = kwargs.get("csa_d_g", 128)
        self.csa_rope_dim = kwargs.get("csa_rope_dim", 64)
        self.hca_m_prime = kwargs.get("hca_m_prime", 32)
        self.n_win = kwargs.get("n_win", 64)
        self.attention_layers = kwargs.get("attention_layers", "interleaved")

    def get_attention_type(self, layer_id):
        if not self.use_csa_hca:
            return "full"
        if self.attention_layers == "interleaved":
            if layer_id < 2:
                return "full"
            return "csa" if layer_id % 2 == 0 else "hca"
        return self.attention_layers

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

def apply_partial_rope(x, cos, sin, rope_dim):
    first, last = x[..., :-rope_dim], x[..., -rope_dim:]
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)
    for _ in range(x.dim() - 3):
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    cos_ = cos[..., :rope_dim]
    sin_ = sin[..., :rope_dim]
    last = ((last * cos_) + (rotate_half(last) * sin_)).to(x.dtype)
    return torch.cat([first, last], dim=-1)

# ============================================================
# CSA: Compressed Sparse Attention (DeepSeek-V4 §2.3.1)
# ============================================================
class CSAAttention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        d = config.hidden_size
        c = config.head_dim if config.num_key_value_heads is not None else config.head_dim
        c = config.head_dim
        n_h = config.num_attention_heads
        self.n_heads = n_h
        self.head_dim = c
        self.is_causal = True

        m = config.csa_m
        self.m = m
        self.top_k = config.csa_top_k
        self.d_c = config.csa_d_c
        self.n_ih = config.csa_n_ih
        self.c_I = config.csa_c_I
        self.g = config.csa_g
        self.d_g = config.csa_d_g
        self.rope_dim = config.csa_rope_dim
        self.n_win = config.n_win

        # Dual-series KV compression
        self.W_aKV = nn.Linear(d, c, bias=False)
        self.W_aZ  = nn.Linear(d, c, bias=False)
        self.W_bKV = nn.Linear(d, c, bias=False)
        self.W_bZ  = nn.Linear(d, c, bias=False)
        self.Ba = nn.Parameter(torch.zeros(1, m, c))
        self.Bb = nn.Parameter(torch.zeros(1, m, c))

        # Low-rank queries
        self.W_DQ = nn.Linear(d, self.d_c, bias=False)
        self.W_UQ = nn.Linear(self.d_c, n_h * c, bias=False)

        # Lightning Indexer
        self.W_IUQ = nn.Linear(self.d_c, self.n_ih * self.c_I, bias=False)
        self.W_w = nn.Linear(d, self.n_ih, bias=False)
        self.W_aKVI = nn.Linear(d, self.c_I, bias=False)
        self.W_aZI  = nn.Linear(d, self.c_I, bias=False)
        self.W_bKVI = nn.Linear(d, self.c_I, bias=False)
        self.W_bZI  = nn.Linear(d, self.c_I, bias=False)
        self.BaI = nn.Parameter(torch.zeros(1, m, self.c_I))
        self.BbI = nn.Parameter(torch.zeros(1, m, self.c_I))

        # Sliding window raw KV projection
        self.W_winKV = nn.Linear(d, c, bias=False)

        # Grouped output projection
        heads_per_group = n_h // self.g
        self.group_projs = nn.ModuleList([
            nn.Linear(heads_per_group * c, self.d_g, bias=False) for _ in range(self.g)
        ])
        self.final_proj = nn.Linear(self.g * self.d_g, d, bias=False)

        # Norms
        self.kv_norm = RMSNorm(c, eps=config.rms_norm_eps)
        self.q_norm = RMSNorm(c, eps=config.rms_norm_eps)

        # Attention sink
        self.sink_logits = nn.Parameter(torch.zeros(n_h))

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        B, N, d = x.shape
        cos, sin = position_embeddings

        # --- Queries (low-rank) ---
        c_q = self.W_DQ(x)
        q = self.W_UQ(c_q).reshape(B, N, self.n_heads, self.head_dim)
        q = apply_partial_rope(q, cos, sin, self.rope_dim)
        q = self.q_norm(q)  # (B, N, n_h, c)
        q_t = q.transpose(1, 2)  # (B, n_h, N, c)

        # --- KV compression (dual-series, overlapped) ---
        C_a = self.W_aKV(x)
        Z_a = self.W_aZ(x)
        C_b = self.W_bKV(x)
        Z_b = self.W_bZ(x)

        n_blocks = N // self.m
        compressed_kv, block_pos = self._compress_dual(C_a, C_b, Z_a, Z_b, n_blocks)
        # compressed_kv: (B, n_blocks, c), block_pos: centers of each block

        # Partial RoPE + norm on compressed KV
        bcos = cos[:n_blocks * self.m:self.m] if n_blocks > 0 else cos.new_zeros(0, cos.size(-1))
        bsin = sin[:n_blocks * self.m:self.m] if n_blocks > 0 else sin.new_zeros(0, sin.size(-1))
        if n_blocks > 0:
            compressed_kv = apply_partial_rope(compressed_kv, bcos, bsin, self.rope_dim)
        compressed_kv = self.kv_norm(compressed_kv)  # (B, n_blocks, c)
        # Expand for MQA: (B, n_blocks, 1, c)
        compressed_kv = compressed_kv.unsqueeze(2)

        # --- Lightning Indexer ---
        if n_blocks > 0 and self.top_k < n_blocks:
            K_IComp, _ = self._compress_indexer_dual(x, n_blocks)
            q_I = self.W_IUQ(c_q).reshape(B, N, self.n_ih, self.c_I)
            w_I = self.W_w(x)  # (B, N, n_ih)

            index_scores = self._compute_index_scores(q_I, w_I, K_IComp, n_blocks)
            # index_scores: (B, n_head=N as queries, n_blocks)
            top_k_idx = torch.topk(index_scores, min(self.top_k, n_blocks), dim=-1).indices
        else:
            top_k_idx = None

        # --- Sliding window raw KV ---
        n_win = min(self.n_win, N)
        win_start = N - n_win
        C_win = self.W_winKV(x[:, win_start:, :])  # (B, n_win, c)
        win_pos = cos[win_start:]

        C_win = apply_partial_rope(C_win, win_pos, sin[win_start:], self.rope_dim)
        C_win = self.kv_norm(C_win).unsqueeze(2)  # (B, n_win, 1, c)

        # --- Assemble K,V and do attention ---
        output = self._core_attention(q_t, compressed_kv, top_k_idx, C_win, n_blocks)
        output = output.transpose(1, 2).reshape(B, N, -1)

        # --- Grouped output projection ---
        output = self._grouped_output(output)
        output = self.resid_dropout(output)
        return output, None

    def _compress_dual(self, C_a, C_b, Z_a, Z_b, n_blocks):
        B, N, c = C_a.shape
        m = self.m
        if n_blocks == 0:
            return C_a.new_zeros(B, 0, c), C_a.new_zeros(0)

        # Build window indices for all blocks at once
        # block i uses C_a[i*m:(i+1)*m] and C_b[max(0,(i-1)*m):i*m]
        idx_a = torch.arange(n_blocks, device=C_a.device)[:, None] * m + torch.arange(m, device=C_a.device)  # (n_blocks, m)
        idx_b = idx_a - m  # (n_blocks, m); idx_b[0] has -m..-1, others have (i-1)*m..i*m-1

        # Gather piece_a for all blocks: (B, n_blocks, m, c)
        piece_a = C_a[:, idx_a, :]  # (B, n_blocks, m, c)
        z_a = Z_a[:, idx_a, :] + self.Ba.unsqueeze(0)

        # Gather piece_b: handle negative indices for block 0
        pad = torch.zeros(B, 1, m, c, device=C_a.device)
        z_pad = torch.full((B, 1, m, c), float('-inf'), device=Z_a.device)
        idx_b_clamped = idx_b.clamp(min=0)  # for block 0, all become 0; we'll mask them
        piece_b = C_b[:, idx_b_clamped, :]  # (B, n_blocks, m, c)
        z_b = Z_b[:, idx_b_clamped, :] + self.Bb.unsqueeze(0)
        # Mask block 0's invalid entries (where idx_b < 0)
        mask_b = (idx_b < 0).unsqueeze(0).unsqueeze(-1)  # (1, n_blocks, m, 1)
        piece_b = torch.where(mask_b, pad, piece_b)
        z_b = torch.where(mask_b, z_pad, z_b)

        # Concatenate along block dim: (B, n_blocks, 2m, c)
        z = torch.cat([z_a, z_b], dim=2)
        c_combined = torch.cat([piece_a, piece_b], dim=2)

        scores = F.softmax(z, dim=2)
        result = (scores * c_combined).sum(dim=2)
        return result, C_a.new_zeros(n_blocks)

    def _compress_indexer_dual(self, x, n_blocks):
        C_a = self.W_aKVI(x)
        Z_a = self.W_aZI(x)
        C_b = self.W_bKVI(x)
        Z_b = self.W_bZI(x)
        B, N, cI = C_a.shape
        m = self.m
        if n_blocks == 0:
            return C_a.new_zeros(B, 0, cI), C_a.new_zeros(0)

        idx_a = torch.arange(n_blocks, device=C_a.device)[:, None] * m + torch.arange(m, device=C_a.device)
        idx_b = idx_a - m

        piece_a = C_a[:, idx_a, :]
        z_a = Z_a[:, idx_a, :] + self.BaI.unsqueeze(0)

        pad = torch.zeros(B, 1, m, cI, device=C_a.device)
        z_pad = torch.full((B, 1, m, cI), float('-inf'), device=Z_a.device)
        idx_b_clamped = idx_b.clamp(min=0)
        piece_b = C_b[:, idx_b_clamped, :]
        z_b = Z_b[:, idx_b_clamped, :] + self.BbI.unsqueeze(0)
        mask_b = (idx_b < 0).unsqueeze(0).unsqueeze(-1)
        piece_b = torch.where(mask_b, pad, piece_b)
        z_b = torch.where(mask_b, z_pad, z_b)

        z = torch.cat([z_a, z_b], dim=2)
        c = torch.cat([piece_a, piece_b], dim=2)
        scores = F.softmax(z, dim=2)
        result = (scores * c).sum(dim=2)
        return result, C_a.new_zeros(n_blocks)

    def _compute_index_scores(self, q_I, w_I, K_IComp, n_blocks):
        B, N, n_ih, cI = q_I.shape
        q_I_t = q_I.transpose(1, 2)
        K_IComp_t = K_IComp.transpose(1, 2).unsqueeze(1)
        raw = torch.matmul(q_I_t, K_IComp_t)
        raw = F.relu(raw)
        w_t = w_I.transpose(1, 2).unsqueeze(-1)
        scores = (raw * w_t).sum(dim=1)
        # Causal mask: query at position t can only attend to blocks before t//m
        t = torch.arange(N, device=scores.device).view(1, -1, 1)
        block_idx = torch.arange(n_blocks, device=scores.device).view(1, 1, -1)
        causal_mask = (block_idx < t // self.m).float()
        # For queries with no preceding blocks, keep scores finite
        has_blocks = causal_mask.sum(dim=-1, keepdim=True) > 0
        scores = scores * causal_mask + (1.0 - causal_mask) * (0.0 - 1e9 * has_blocks.float())
        # When no preceding blocks exist, give uniform near-zero scores
        scores = scores.masked_fill(~has_blocks.expand_as(scores), -1e4)
        return scores

    def _core_attention(self, q, compressed_kv, top_k_idx, win_kv, n_blocks):
        B, n_h, N, c = q.shape
        n_win = win_kv.size(1)
        m = self.m

        blk_pos = torch.arange(m - 1, n_blocks * m, m, device=q.device).long() if n_blocks > 0 else q.new_zeros(0, dtype=torch.long)
        win_pos = torch.arange(N - n_win, N, device=q.device).long() if n_win > 0 else q.new_zeros(0, dtype=torch.long)

        if n_blocks == 0 or top_k_idx is None or self.top_k >= n_blocks:
            kv_cat = torch.cat([compressed_kv.squeeze(2), win_kv.squeeze(2)], dim=1)
            kv_cat = kv_cat.unsqueeze(1).expand(-1, n_h, -1, -1)
            kv_pos = torch.cat([blk_pos, win_pos])
            scores = (q @ kv_cat.transpose(-2, -1)) / math.sqrt(c)
            q_pos = torch.arange(N, device=q.device)
            mask = kv_pos.unsqueeze(0) >= q_pos.unsqueeze(1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            exp_scores = torch.exp(scores - scores.max(dim=-1, keepdim=True).values.clamp(min=0))
            sink = torch.exp(self.sink_logits.view(1, -1, 1, 1).to(exp_scores.dtype))
            attn = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + sink)
            return attn @ kv_cat

        top_k = min(self.top_k, n_blocks)
        kv_flat = compressed_kv.squeeze(2).unsqueeze(1).expand(-1, N, -1, -1)
        kv_flat = kv_flat.reshape(-1, compressed_kv.size(1), c)
        flat_idx = top_k_idx[:, :, :top_k].reshape(B * N, top_k)
        flat_idx_exp = flat_idx.unsqueeze(-1).expand(-1, -1, c)
        gathered = torch.gather(kv_flat, 1, flat_idx_exp).reshape(B, N, top_k, c)

        win_expanded = win_kv.squeeze(2).unsqueeze(1).expand(-1, N, -1, -1)
        kv_cat = torch.cat([gathered, win_expanded], dim=2)
        K_len = kv_cat.size(2)

        kv_exp = kv_cat.unsqueeze(1).expand(-1, n_h, -1, -1, -1)
        q_exp = q.unsqueeze(3)
        scores = (q_exp @ kv_exp.transpose(-2, -1)).squeeze(3) / math.sqrt(c)

        # Causal mask for gathered blocks
        blk_end_pos = top_k_idx[:, :, :top_k] * m + (m - 1)
        t_idx = torch.arange(N, device=q.device).view(1, -1, 1)
        blk_mask = blk_end_pos >= t_idx
        scores[:, :, :, :top_k] = scores[:, :, :, :top_k].masked_fill(
            blk_mask.unsqueeze(1), float('-inf'))

        if n_win > 0:
            t_pos = torch.arange(N, device=q.device)[:, None]
            win_start = N - n_win
            w_pos = (torch.arange(n_win, device=q.device) + win_start)[None, :]
            win_mask = w_pos >= t_pos
            scores[:, :, :, -n_win:] = scores[:, :, :, -n_win:].masked_fill(
                win_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        exp_scores = torch.exp(scores - scores.max(dim=-1, keepdim=True).values.clamp(min=0))
        sink = torch.exp(self.sink_logits.view(1, -1, 1, 1).to(exp_scores.dtype))
        attn = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + sink)
        return (attn.unsqueeze(-2) @ kv_exp).squeeze(-2)

    def _grouped_output(self, x):
        B, N, D = x.shape
        heads_per_group = self.n_heads // self.g
        groups = []
        for i in range(self.g):
            si = i * heads_per_group * self.head_dim
            ei = (i + 1) * heads_per_group * self.head_dim
            group = self.group_projs[i](x[:, :, si:ei])
            groups.append(group)
        concat = torch.cat(groups, dim=-1)
        return self.final_proj(concat)


# ============================================================
# HCA: Heavily Compressed Attention (DeepSeek-V4 §2.3.2)
# ============================================================
class HCAAttention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        d = config.hidden_size
        c = config.head_dim
        n_h = config.num_attention_heads
        self.n_heads = n_h
        self.head_dim = c
        self.is_causal = True

        self.m_prime = config.hca_m_prime
        self.d_c = config.csa_d_c
        self.csa_g = config.csa_g
        self.d_g = config.csa_d_g
        self.g = config.csa_g
        self.rope_dim = config.csa_rope_dim
        self.n_win = config.n_win

        # Single-series KV compression
        self.W_KV = nn.Linear(d, c, bias=False)
        self.W_Z  = nn.Linear(d, c, bias=False)
        self.B    = nn.Parameter(torch.zeros(1, self.m_prime, c))

        # Low-rank queries
        self.W_DQ = nn.Linear(d, self.d_c, bias=False)
        self.W_UQ = nn.Linear(self.d_c, n_h * c, bias=False)

        # Sliding window raw KV
        self.W_winKV = nn.Linear(d, c, bias=False)

        # Grouped output projection
        heads_per_group = n_h // self.g
        self.group_projs = nn.ModuleList([
            nn.Linear(heads_per_group * c, self.d_g, bias=False) for _ in range(self.g)
        ])
        self.final_proj = nn.Linear(self.g * self.d_g, d, bias=False)

        # Norms
        self.kv_norm = RMSNorm(c, eps=config.rms_norm_eps)
        self.q_norm = RMSNorm(c, eps=config.rms_norm_eps)

        # Attention sink
        self.sink_logits = nn.Parameter(torch.zeros(n_h))

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        B, N, d = x.shape
        cos, sin = position_embeddings

        # Queries
        c_q = self.W_DQ(x)
        q = self.W_UQ(c_q).reshape(B, N, self.n_heads, self.head_dim)
        q = apply_partial_rope(q, cos, sin, self.rope_dim)
        q = self.q_norm(q)
        q_t = q.transpose(1, 2)

        # Single-series KV compression
        KV = self.W_KV(x)
        Z = self.W_Z(x)
        m_prime = self.m_prime
        n_blocks = N // m_prime
        if n_blocks > 0:
            idx = torch.arange(n_blocks, device=q.device)[:, None] * m_prime + torch.arange(m_prime, device=q.device)
            windows = KV[:, idx, :]
            z = Z[:, idx, :] + self.B.unsqueeze(0)
            scores = F.softmax(z, dim=2)
            compressed_kv = (scores * windows).sum(dim=2)
        else:
            compressed_kv = KV.new_zeros(B, 0, self.head_dim)

        # Partial RoPE + norm
        if n_blocks > 0:
            bcos = cos[:n_blocks * m_prime:m_prime]
            bsin = sin[:n_blocks * m_prime:m_prime]
            compressed_kv = apply_partial_rope(compressed_kv, bcos, bsin, self.rope_dim)
        compressed_kv = self.kv_norm(compressed_kv).unsqueeze(2)  # (B, n_blocks, 1, c)

        # Sliding window
        n_win = min(self.n_win, N)
        C_win = self.W_winKV(x[:, N - n_win:, :])
        win_pos = cos[N - n_win:]
        C_win = apply_partial_rope(C_win, win_pos, sin[N - n_win:], self.rope_dim)
        C_win = self.kv_norm(C_win).unsqueeze(2)

        # Dense attention on compressed KV + sliding window
        kv = compressed_kv.squeeze(2)  # (B, n_blocks, c)
        win_kv_hca = C_win.squeeze(2)  # (B, n_win, c)
        kv_cat = torch.cat([kv, win_kv_hca], dim=1)
        kv_cat = kv_cat.unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        # Position-based causal mask
        blk_pos = torch.arange(self.m_prime - 1, n_blocks * self.m_prime, self.m_prime, device=q.device).long() if n_blocks > 0 else q.new_zeros(0, dtype=torch.long)
        win_pos = torch.arange(N - n_win + 1, N + 1, device=q.device).long() - 1 if n_win > 0 else q.new_zeros(0, dtype=torch.long)
        kv_pos = torch.cat([blk_pos, win_pos])

        scores = (q_t @ kv_cat.transpose(-2, -1)) / math.sqrt(self.head_dim)
        q_pos = torch.arange(N, device=q.device)
        mask = kv_pos.unsqueeze(0) >= q_pos.unsqueeze(1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        exp_scores = torch.exp(scores - scores.max(dim=-1, keepdim=True).values.clamp(min=0))
        sink = self.sink_logits.view(1, -1, 1, 1).to(exp_scores.dtype)
        attn = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + torch.exp(sink))
        out = attn @ kv_cat
        out = out.transpose(1, 2).reshape(B, N, -1)

        # Grouped output
        out = self._grouped_output(out)
        out = self.resid_dropout(out)
        return out, None

    def _grouped_output(self, x):
        B, N, D = x.shape
        heads_per_group = self.n_heads // self.g
        groups = []
        for i in range(self.g):
            si = i * heads_per_group * self.head_dim
            ei = (i + 1) * heads_per_group * self.head_dim
            groups.append(self.group_projs[i](x[:, :, si:ei]))
        return self.final_proj(torch.cat(groups, dim=-1))
class MHCBlock(nn.Module):
    """
    替换标准残差连接: X_{l+1} = B_l @ X_l + C_l * F_l(A_l @ X_l)
    B_l 被约束在双随机矩阵流形上（Sinkhorn-Knopp），谱范数 ≤ 1，确保梯度传播稳定。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        n_hc = config.mhc_n_hc
        d = config.hidden_size
        D = n_hc * d
        self.n_hc = n_hc
        self.sinkhorn_iters = config.mhc_sinkhorn_iters

        self.W_pre = nn.Parameter(torch.randn(D, n_hc) * 0.02)
        self.S_pre = nn.Parameter(torch.zeros(1, n_hc))
        self.alpha_pre = nn.Parameter(torch.zeros(1))

        self.W_res = nn.Parameter(torch.randn(D, n_hc * n_hc) * 0.02)
        self.S_res = nn.Parameter(torch.zeros(n_hc, n_hc))
        self.alpha_res = nn.Parameter(torch.zeros(1))

        self.W_post = nn.Parameter(torch.randn(D, 1) * 0.02)
        self.S_post = nn.Parameter(torch.zeros(n_hc, 1))
        self.alpha_post = nn.Parameter(torch.zeros(1))

    def forward(self, X, layer_fn):
        B, N, n_hc, d = X.shape
        X_flat = X.reshape(B, N, -1)
        X_hat = F.rms_norm(X_flat.float(), normalized_shape=(X_flat.size(-1),)).to(X.dtype)

        A = torch.sigmoid(self.alpha_pre * (X_hat @ self.W_pre) + self.S_pre)
        layer_input = torch.einsum('bnk,bnkd->bnd', A, X)
        layer_output, present = layer_fn(layer_input)

        B_raw = self.alpha_res * (X_hat @ self.W_res).reshape(B, N, n_hc, n_hc) + self.S_res
        B_mat = self._sinkhorn_knopp(B_raw, self.sinkhorn_iters)

        C = 2.0 * torch.sigmoid(self.alpha_post * (X_hat @ self.W_post).unsqueeze(-1) + self.S_post)

        X_next = torch.matmul(B_mat, X) + C * layer_output.unsqueeze(2)
        return X_next, present

    def _sinkhorn_knopp(self, M_raw, num_iters):
        M = M_raw.clone()
        M = M.view(M.size(0), M.size(1), -1)
        mx = M.max(dim=-1, keepdim=True).values.unsqueeze(-1)
        M = torch.exp(M_raw - mx)
        for _ in range(num_iters):
            M = M / M.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            M = M / M.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        return M


class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        attn_type = config.get_attention_type(layer_id)
        if attn_type == "csa":
            self.self_attn = CSAAttention(config)
        elif attn_type == "hca":
            self.self_attn = HCAAttention(config)
        else:
            self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mhc_blocks = nn.ModuleList([MHCBlock(config) for _ in range(self.num_hidden_layers)]) if config.use_mhc else None
        self.mhc_n_hc = config.mhc_n_hc if config.use_mhc else 0
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []

        if self.config.use_mhc:
            X = hidden_states.unsqueeze(2).expand(-1, -1, self.mhc_n_hc, -1)
            for layer, mhc, past_key_value in zip(self.layers, self.mhc_blocks, past_key_values):
                def layer_fn(x, l=layer, pe=position_embeddings, pkv=past_key_value, uc=use_cache, am=attention_mask):
                    return l(x, pe, past_key_value=pkv, use_cache=uc, attention_mask=am)
                X, present = mhc(X, layer_fn)
                presents.append(present)
            hidden_states = X.mean(dim=2)
        else:
            for layer, past_key_value in zip(self.layers, past_key_values):
                hidden_states, present = layer(
                    hidden_states, position_embeddings,
                    past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask
                )
                presents.append(present)

        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings: self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
    
    # https://github.com/jingyaogong/minimind/discussions/611
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids