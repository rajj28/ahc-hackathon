"""Trainable heads on top of frozen embeddings. Small by design: the whole point is that a
train/evaluate/change cycle takes minutes, not hours.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipHead(nn.Module):
    """Level 1. [T,D] -> logits[12]  (11 anomaly classes + normal at index 11).

    Attention pooling over frames, not mean pooling: L1 clips run 5.7-26s and the evidence is
    often in a few frames (an accident, a fight), which mean pooling dilutes.
    Target ~0.5M params.
    """
    def __init__(self, d_in: int, cfg):
        super().__init__()
        hidden = cfg.model.clip_head.hidden
        self.project = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                     nn.Dropout(cfg.model.clip_head.dropout))
        self.attention = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(hidden, 12)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        single_clip = x.ndim == 2
        if single_clip:
            x = x.unsqueeze(0)
        features = self.project(x)
        attention = self.attention(features).squeeze(-1)
        if mask is not None:
            attention = attention.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(attention, dim=1).unsqueeze(-1)
        logits = self.classifier((features * weights).sum(dim=1))
        return logits.squeeze(0) if single_clip else logits


class TemporalHead(nn.Module):
    """Levels 2/3. [T,D] -> (frame_logits[T,12], boundary[T]).

      Conv1d stem (kernel 5, d_model 256) for local temporal context
      -> 2-layer TransformerEncoder (4 heads, dropout 0.1) for long-range context
      -> linear class head + linear boundary head

    Must accept variable T without retraining: T ranges from ~11 frames (5.7s @ 2fps) to
    ~1258 (628.8s @ 2fps). Use sinusoidal position encoding or none at all - never a learned
    fixed-length table.

    Input is the L2-normalized embedding concatenated with its 1-frame delta when
    cfg.model.temporal_head.use_delta: motion matters for accident/fighting/wrong_way, and the
    delta is free to compute.

    The boundary head is only trainable because SpliceDataset manufactures exact splice points
    - the raw train timestamps have no boundary signal for 7 of 11 classes.
    Target ~3-6M params.
    """
    def __init__(self, d_in: int, cfg):
        super().__init__()
        self.use_delta = bool(cfg.model.temporal_head.use_delta)
        width = d_in * (2 if self.use_delta else 1)
        d_model = cfg.model.temporal_head.d_model
        self.stem = nn.Conv1d(width, d_model, cfg.model.temporal_head.conv_kernel,
                              padding=cfg.model.temporal_head.conv_kernel // 2)
        layer = nn.TransformerEncoderLayer(d_model, cfg.model.temporal_head.heads,
                                           dim_feedforward=d_model * 4,
                                           dropout=cfg.model.temporal_head.dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.model.temporal_head.layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier, self.boundary = nn.Linear(d_model, 12), nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor):
        single = x.ndim == 2
        if single: x = x.unsqueeze(0)
        x = F.normalize(x, dim=-1)
        if self.use_delta:
            delta = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
            x = torch.cat((x, delta), dim=-1)
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)
        length, dim = x.shape[1], x.shape[2]
        position = torch.arange(length, device=x.device, dtype=x.dtype).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dim, 2, device=x.device, dtype=x.dtype) *
                          (-torch.log(torch.tensor(10000.0, device=x.device, dtype=x.dtype)) / dim))
        pe = torch.zeros(length, dim, device=x.device, dtype=x.dtype)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        x = self.norm(self.encoder(x + pe.unsqueeze(0)))
        logits, boundary = self.classifier(x), self.boundary(x).squeeze(-1)
        return (logits.squeeze(0), boundary.squeeze(0)) if single else (logits, boundary)


def class_weights(counts: dict, mode: str = "inv_sqrt") -> torch.Tensor:
    """1/sqrt(count), normalized to mean 1. Counts run fire=77 to traffic_accident=565;
    unweighted CE would simply ignore the tail classes.
    """
    if mode != "inv_sqrt":
        raise ValueError(f"unsupported class-weight mode: {mode}")
    values = torch.tensor([1.0 / float(counts.get(index, 1)) ** 0.5 for index in range(12)])
    return values / values.mean()


def losses(frame_logits, boundary, labels, boundary_target, weights, cfg):
    """Class-weighted cross-entropy + BCE on boundary, mixed by cfg.train.boundary_weight.
    -> (total, {'ce':…, 'bnd':…}) for logging.
    """
    ce = F.cross_entropy(frame_logits.reshape(-1, 12), labels.reshape(-1), weight=weights)
    bnd = F.binary_cross_entropy_with_logits(boundary, boundary_target)
    total = ce + cfg.train.boundary_weight * bnd
    return total, {"ce": float(ce.detach()), "bnd": float(bnd.detach())}
