"""_encoder_njt (varlen/jagged attention) must equal the padded+mask encoder: same weights,
same tokens, pad slots excluded either way. CUDA-only (jagged SDPA); skipped on CPU boxes.
Interspersed pad patterns mirror the real token grid (pad INSIDE the state region + trailing
option slots), plus grad-flow and full-logits_value equivalence on a tiny TokenEncoder-shaped
net."""
import pytest
import torch

from rl.policy import TokenTransformer  # noqa: F401  (import check even when skipped)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="jagged SDPA needs CUDA")


def _mini_encoder(d=64, h=2, layers=3, seed=0):
    torch.manual_seed(seed)
    lay = torch.nn.TransformerEncoderLayer(d, h, 4 * d, batch_first=True, dropout=0.0)
    return torch.nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)


class _Shim(torch.nn.Module):
    """Just enough of TokenEncoder to borrow _encoder_njt."""

    def __init__(self, enc):
        super().__init__()
        self.encoder = enc
        self.varlen_attn = True
        self.varlen_compile = False
        self._vlc_qkv = None
        self._vlc_post = None

    _encoder_njt = TokenTransformer._encoder_njt


@cuda
def test_njt_equals_padded_masked():
    torch.manual_seed(1)
    d = 64
    enc = _mini_encoder(d=d).cuda().eval()
    shim = _Shim(enc).cuda().eval()
    B, S = 7, 90
    seq = torch.randn(B, S, d, device="cuda")
    pad = torch.rand(B, S, device="cuda") > 0.6          # interspersed pad
    pad[:, 0] = False                                     # CLS-like: never padded
    ref = enc(seq, src_key_padding_mask=pad)
    out = shim._encoder_njt(seq, pad)
    keep = ~pad
    diff = (ref[keep] - out[keep]).abs().max().item()
    assert diff < 2e-4, f"max |padded - njt| on present tokens = {diff}"
    assert out[pad].abs().max().item() == 0.0             # pad slots scatter back as zeros


@cuda
def test_njt_grads_flow():
    d = 64
    enc = _mini_encoder(d=d).cuda()
    shim = _Shim(enc).cuda()
    seq = torch.randn(3, 40, d, device="cuda", requires_grad=True)
    pad = torch.zeros(3, 40, dtype=torch.bool, device="cuda")
    pad[:, 25:] = True
    shim._encoder_njt(seq, pad).sum().backward()
    assert seq.grad is not None and torch.isfinite(seq.grad).all()
    g = [p.grad for p in enc.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(t).all() for t in g)


@cuda
def test_logits_value_flag_equivalence():
    """Full TokenEncoder forward: varlen_attn on/off must give the same logits/value."""
    from rl.card_features import get_card_table
    from rl.encoding import TokenEncoder as ObsEnc
    from rl.env_selfplay import TwoSidedSelfPlayEnv
    from rl.decks import DECKS
    from rl.policy import build_token_net, obs_to_tensors

    ct = get_card_table()
    net = build_token_net(ct, {"d_model": 64, "nhead": 2, "nlayers": 2, "ff": 64,
                               "static": True, "split_heads": True}).cuda().eval()
    env = TwoSidedSelfPlayEnv(decks=list(DECKS.values())[:3], seed=4,
                              encoder=ObsEnc(ct))
    import random as _r
    import numpy as np
    rng = _r.Random(2)
    obs_batch = []
    enc_o, _s, _ = env.reset()
    for _ in range(48):
        obs_batch.append(enc_o)
        legal = np.flatnonzero(np.asarray(enc_o["action_mask"]) > 0.5)
        enc_o, _s, _r2, done, _ = env.step(int(rng.choice(legal)))
        if done:
            enc_o, _s, _ = env.reset()
    env.close()
    o = obs_to_tensors(obs_batch, net.int_keys if hasattr(net, "int_keys") else None, "cuda") \
        if False else {k: torch.stack([torch.as_tensor(np.asarray(ob[k])) for ob in obs_batch]).cuda()
                       for k in obs_batch[0]}
    with torch.no_grad():
        net.varlen_attn = False
        l0, v0 = net.logits_value(o)
        net.varlen_attn = True
        l1, v1 = net.logits_value(o)
    live = l0 > -1e8                                       # compare only unmasked logits
    dl = (l0[live] - l1[live]).abs().max().item()
    dv = (v0 - v1).abs().max().item()
    assert dl < 2e-3, f"logit diff {dl}"
    assert dv < 2e-3, f"value diff {dv}"
