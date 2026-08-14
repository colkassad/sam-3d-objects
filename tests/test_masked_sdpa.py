import torch
import torch.nn.functional as F

from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.attention.masked_sdpa import (
    block_diag_attn_mask,
    masked_sdpa,
)


def legacy_masked_sdpa(q, k, v, q_seqlen, kv_seqlen):
    mask = block_diag_attn_mask(q_seqlen, kv_seqlen, q.device, q.dtype)
    out = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        attn_mask=mask[None, None],
        dropout_p=0.0,
    )
    return out.permute(0, 2, 1, 3)[0]


@torch.inference_mode()
def test_single_sequence_matches_unmasked_sdpa():
    q = torch.randn(1, 11, 2, 8)
    k = torch.randn(1, 13, 2, 8)
    v = torch.randn(1, 13, 2, 8)

    actual = masked_sdpa(q, k, v, [11], [13])
    expected = legacy_masked_sdpa(q, k, v, [11], [13])

    torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def test_multiple_sequences_match_block_diagonal_mask():
    q_lengths = [3, 5, 2]
    kv_lengths = [4, 2, 6]
    q = torch.randn(1, sum(q_lengths), 2, 8)
    k = torch.randn(1, sum(kv_lengths), 2, 8)
    v = torch.randn(1, sum(kv_lengths), 2, 8)

    actual = masked_sdpa(q, k, v, q_lengths, kv_lengths)
    expected = legacy_masked_sdpa(q, k, v, q_lengths, kv_lengths)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
