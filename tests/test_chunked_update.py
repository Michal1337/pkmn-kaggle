"""INT32-safe chunked update: the sum-form per-chunk loss contributions (train_selfplay update_net)
must accumulate to EXACTLY the whole-minibatch gradients. Mirrors the trainer's loss formulas on a
tiny policy+value head; compares grads whole-vs-2-chunks for both branches (standard 2-side and
finetune agent-masked) incl. teacher KL/value terms and full-minibatch adv normalization."""
import torch
from torch.distributions import Categorical

torch.manual_seed(0)

N, A, D = 64, 7, 12          # rows, actions, feature dim
CLIP, ENT_C, VF_C, TKL, TVD = 0.2, 0.01, 0.5, 0.005, 0.005


def make_net():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(D, 32), torch.nn.Tanh(),
                               torch.nn.Linear(32, A + 1))     # logits + value


def fwd(net, x):
    out = net(x)
    return out[:, :A], out[:, A]


def loss_rows(net, data, rows, adv_full, msum, n_rows, m_full):
    x, actions, old_logp, returns, t_logits, t_val = (data[k] for k in
                                                      ("x", "act", "logp", "ret", "tl", "tv"))
    s_logits, newval = fwd(net, x[rows])
    pdist = Categorical(logits=s_logits)
    newlogp = pdist.log_prob(actions[rows]); entropy = pdist.entropy()
    ratio = (newlogp - old_logp[rows]).exp()
    adv = adv_full[rows]
    pgmax = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - CLIP, 1 + CLIP))
    if m_full is None:
        pg = pgmax.sum() / n_rows
        ent = entropy.sum() / n_rows
    else:
        m = m_full[rows]
        pg = (pgmax * m).sum() / msum
        ent = (entropy * m).sum() / msum
    v = 0.5 * ((newval - returns[rows]) ** 2).sum() / n_rows
    t_logp = torch.log_softmax(t_logits[rows], dim=-1)
    s_logp = torch.log_softmax(s_logits, dim=-1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).sum() / n_rows
    vd = 0.5 * ((newval - t_val[rows]) ** 2).sum() / n_rows
    return pg - ENT_C * ent + VF_C * v + TKL * kl + TVD * vd


def grads(net, data, chunks, mask):
    n_rows = float(N)
    adv_full = data["adv"].clone()
    m_full = msum = None
    if mask:
        m_full = data["mask"]; msum = m_full.sum().clamp(min=1.0)
        mu = (adv_full * m_full).sum() / msum
        sd = (((adv_full - mu) ** 2 * m_full).sum() / msum).sqrt()
        adv_full = (adv_full - mu) / (sd + 1e-8)
    else:
        adv_full = (adv_full - adv_full.mean()) / (adv_full.std() + 1e-8)
    net.zero_grad()
    csz = (N + chunks - 1) // chunks
    for ci in range(0, N, csz):
        rows = torch.arange(ci, min(ci + csz, N))
        loss_rows(net, data, rows, adv_full, msum, n_rows, m_full).backward()
    return [p.grad.clone() for p in net.parameters()]


def _data():
    torch.manual_seed(2)
    return {"x": torch.randn(N, D), "act": torch.randint(0, A, (N,)),
            "logp": -torch.rand(N) * 2, "ret": torch.randn(N), "adv": torch.randn(N),
            "mask": (torch.rand(N) > 0.5).float(),
            "tl": torch.randn(N, A), "tv": torch.randn(N)}


def test_chunked_grads_standard():
    data = _data()
    g1 = grads(make_net(), data, chunks=1, mask=False)
    g2 = grads(make_net(), data, chunks=2, mask=False)
    g4 = grads(make_net(), data, chunks=4, mask=False)
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
    for a, b in zip(g1, g4):
        assert torch.allclose(a, b, atol=1e-6)


def test_chunked_grads_finetune_masked():
    data = _data()
    g1 = grads(make_net(), data, chunks=1, mask=True)
    g3 = grads(make_net(), data, chunks=3, mask=True)
    for a, b in zip(g1, g3):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()
