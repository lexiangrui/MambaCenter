"""Selective-scan backend and scan routes shared by the VSS, CP-SSM and IA-SSM mixers.

Every mixer uses the native Mamba recurrence of ``mamba_ssm.selective_scan_fn``::

    h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t u_t,      y_t = C_t h_t + D u_t

Kernel layouts: u, Δ (b, d, l); B, C (b, g, n, l) with d % g == 0 (one group per
route); A (d, n); D and the Δ bias (d,). The CUDA kernel runs on GPU; the torch
reference implements the same recurrence for CPU tests and kernel checks.
"""
import torch
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan
    CUDA_IMPORT_ERROR = None
except Exception as exc:  # CPU-only machines fall back to the reference recurrence
    _cuda_scan = None
    CUDA_IMPORT_ERROR = repr(exc)

# Row-major, column-major and their reversals (VMamba cross-scan order).
ROUTES = ('h', 'v', 'h_rev', 'v_rev')


def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False,
                   backend='auto'):
    """Run the selective scan; returns y with the shape and dtype of ``u``."""
    if backend == 'auto':
        backend = 'cuda' if u.is_cuda and _cuda_scan is not None else 'torch'
    if backend == 'cuda':
        if _cuda_scan is None:
            raise RuntimeError('mamba_ssm selective scan unavailable: %s' % CUDA_IMPORT_ERROR)
        return _cuda_scan(u.contiguous(), delta.contiguous(), A.contiguous(), B.contiguous(),
                          C.contiguous(), D, None, delta_bias, delta_softplus)
    if backend != 'torch':
        raise ValueError('unknown scan backend: %s' % backend)
    return selective_scan_ref(u, delta, A, B, C, D, delta_bias, delta_softplus)


def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    """Sequential fp32 recurrence with grouped B/C, identical in meaning to the kernel."""
    dtype = u.dtype
    u, delta, A = u.float(), delta.float(), A.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    b, d, length = u.shape
    if B.dim() == 3:
        B, C = B.unsqueeze(1), C.unsqueeze(1)
    groups = B.shape[1]
    B = B.float().repeat_interleave(d // groups, dim=1)      # (b, d, n, l)
    C = C.float().repeat_interleave(d // groups, dim=1)
    decay = torch.exp(delta.unsqueeze(2) * A[None, :, :, None])
    drive = delta.unsqueeze(2) * B * u.unsqueeze(2)
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for t in range(length):
        h = decay[..., t] * h + drive[..., t]
        ys.append((h * C[..., t]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D.float()[None, :, None]
    return y.to(dtype)


def flatten_routes(x, routes):
    """(b, c, h, w) -> (b, k, c, h*w): one token sequence per route."""
    seqs = []
    for route in routes:
        seq = x.transpose(2, 3).flatten(2) if route.startswith('v') else x.flatten(2)
        seqs.append(seq.flip(-1) if route.endswith('_rev') else seq)
    return torch.stack(seqs, dim=1)


def restore_routes(seqs, routes, h, w):
    """(b, k, c, h*w) -> (b, k, c, h, w): inverse of flatten_routes for each route."""
    maps = []
    for i, route in enumerate(routes):
        seq = seqs[:, i]
        if route.endswith('_rev'):
            seq = seq.flip(-1)
        if route.startswith('v'):
            maps.append(seq.unflatten(-1, (w, h)).transpose(-1, -2))
        else:
            maps.append(seq.unflatten(-1, (h, w)))
    return torch.stack(maps, dim=1)


def _priority_key_order(score, bins, descending):
    """Stable order of a raster-ordered score row: by prior level, raster order inside."""
    if bins is None:
        return score.argsort(dim=1, descending=descending, stable=True)
    level = (score * bins).floor().clamp_(0, bins - 1)
    return level.argsort(dim=1, descending=descending, stable=True)


def priority_order(score, bins=4, descending=False):
    """Permutation (b, l) of raster indices: low prior first, so candidates are read out last.

    score (b, l) in [0, 1] is quantised into ``bins`` levels; tokens keep raster order
    inside a level. ``bins=None`` sorts by the raw score.
    """
    return _priority_key_order(score.detach().float(), bins, descending)


def select_tokens(score, ratio, extra_ratio=0.0, bins=4, descending=False):
    """Raster indices (b, k) of the top-ρN prior tokens plus random extras, in priority order.

    ``extra_ratio`` adds that fraction of the remaining tokens uniformly at random
    (training only). k is identical for every image of the batch.
    """
    score = score.detach().float()
    b, length = score.shape
    k = min(length, max(1, int(round(ratio * length))))
    idx = score.topk(k, dim=1, sorted=False).indices
    extra = min(length - k, int(round(extra_ratio * length)))
    if extra > 0:
        noise = torch.rand(b, length, device=score.device)
        noise.scatter_(1, idx, -1.0)
        idx = torch.cat([idx, noise.topk(extra, dim=1, sorted=False).indices], dim=1)
    idx = idx.sort(dim=1).values
    return idx.gather(1, _priority_key_order(score.gather(1, idx), bins, descending))


def gather_tokens(x, idx):
    """x (b, c, l), idx (b, k) -> (b, c, k)."""
    return x.gather(2, idx.unsqueeze(1).expand(-1, x.shape[1], -1))


def scatter_tokens(values, idx, length):
    """values (b, c, k) placed at idx (b, k) of a zero (b, c, length) sequence."""
    out = values.new_zeros(values.shape[0], values.shape[1], length)
    return out.scatter(2, idx.unsqueeze(1).expand(-1, values.shape[1], -1), values)


def nonneg(param):
    """max(param, 0) with an identity gradient, so a zero-initialised gain stays trainable."""
    return param + (param.clamp(min=0) - param).detach()


def soft_min(x, tau, dim):
    """-τ log Σ exp(-x/τ): a smooth minimum, at most min(x) and at least min(x) - τ log k."""
    return -tau * torch.logsumexp(-x / tau, dim=dim)
