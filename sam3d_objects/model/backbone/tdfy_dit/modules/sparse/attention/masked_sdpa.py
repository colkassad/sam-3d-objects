# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn.functional as F


def block_diag_attn_mask(q_seqlens, kv_seqlens, device=None, dtype=torch.float32):
    """
    Create an additive attention mask for block-diagonal attention.
    The result is shape [sum_q, sum_kv], with 0.0 in the valid
    region(s) and -inf elsewhere.
    """
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    # Start with everything "masked out"
    attn_mask = torch.full(
        (total_q, total_kv), float("-inf"), device=device, dtype=dtype
    )

    q_start = 0
    kv_start = 0
    for q_len, kv_len in zip(q_seqlens, kv_seqlens):
        attn_mask[q_start : q_start + q_len, kv_start : kv_start + kv_len] = 0
        q_start += q_len
        kv_start += kv_len

    return attn_mask


def masked_sdpa(q, k, v, q_seqlen, kv_seqlen):
    """
    Mimic xFormers' memory_efficient_attention using PyTorch 2.0 scaled_dot_product_attention.
    """
    if len(q_seqlen) != len(kv_seqlen):
        raise ValueError("q_seqlen and kv_seqlen must have the same batch length")

    # The public inference path has batch size one. Passing no mask allows
    # PyTorch to select FlashAttention or another fused SDPA implementation.
    if len(q_seqlen) == 1:
        out = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        )
        return out.permute(0, 2, 1, 3)[0]

    # A dense block-diagonal mask scales with sum(q_len) * sum(kv_len).
    # Execute independent sequences instead, keeping peak attention storage
    # proportional to the largest sequence pair.
    outputs = []
    q_offset = 0
    kv_offset = 0
    for q_len, kv_len in zip(q_seqlen, kv_seqlen):
        q_i = q[:, q_offset : q_offset + q_len].permute(0, 2, 1, 3)
        k_i = k[:, kv_offset : kv_offset + kv_len].permute(0, 2, 1, 3)
        v_i = v[:, kv_offset : kv_offset + kv_len].permute(0, 2, 1, 3)
        out_i = F.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(out_i.permute(0, 2, 1, 3))
        q_offset += q_len
        kv_offset += kv_len

    return torch.cat(outputs, dim=1)[0]
