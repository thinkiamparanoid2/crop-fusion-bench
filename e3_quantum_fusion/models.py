"""E3 fusion heads (Spec B) for the controlled E2-vs-E3 comparison.

STUB: layer structure is defined (the architecture is locked); the forward logic
is intentionally left to fill in.

Both heads share the projection and readout, differing only in the 4->4 decision
sub-network:

    shared projection : Linear(1344,128) -> BN -> ReLU -> Dropout(0.1) -> Linear(128,4)
    E2-matched middle : tanh -> Linear(4,4,bias=False) -> tanh        (16 params)
    E3-4 middle       : (tanh*pi) -> 4-qubit VQC                       (16 params)
    shared readout    : Linear(4,25)

Verified totals: E2-matched = E3-4 = 173,073 trainable; post-bottleneck head = 16.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantum_layer import VQCLayer  # noqa: E402

FUSED_DIM = 1344
HIDDEN = 128
BOTTLENECK = 4
NUM_CLASSES = 25
ANGLE_SCALE = 3.141592653589793  # tanh output scaled into [-pi, pi] for angle encoding


class FusionProjection(nn.Module):
    """Shared Spec B projection 1344 -> 128 -> 4 (identical across E2-matched / E3-4)."""

    def __init__(self, fused_dim=FUSED_DIM, hidden=HIDDEN, bottleneck=BOTTLENECK, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns the [B, 4] bottleneck (pre-activation)
        return self.proj(x)


class E2MatchedHead(nn.Module):
    """Fair classical comparator: projection -> tanh -> Linear(4,4,bias=False) -> tanh -> Linear(4,25)."""

    def __init__(self, bottleneck=BOTTLENECK, num_classes=NUM_CLASSES):
        super().__init__()
        self.projection = FusionProjection(bottleneck=bottleneck)
        self.middle = nn.Linear(bottleneck, bottleneck, bias=False)  # 16 params (matched)
        self.readout = nn.Linear(bottleneck, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projection(x)
        z = torch.tanh(z)
        z = torch.tanh(self.middle(z))
        return self.readout(z)

    def post_bottleneck_parameters(self):
        # the matched decision sub-network (excludes shared projection/readout)
        return self.middle.parameters()


class E3HybridHead(nn.Module):
    """Quantum head: projection -> (tanh*pi) -> 4-qubit VQC -> Linear(4,25)."""

    def __init__(self, bottleneck=BOTTLENECK, num_classes=NUM_CLASSES, sim="default.qubit"):
        super().__init__()
        self.projection = FusionProjection(bottleneck=bottleneck)
        self.vqc = VQCLayer(n_qubits=bottleneck, sim=sim)  # 16 params (matched)
        self.readout = nn.Linear(bottleneck, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projection(x)
        z = ANGLE_SCALE * torch.tanh(z)
        z = self.vqc(z)
        return self.readout(z)

    def post_bottleneck_parameters(self):
        return self.vqc.parameters()


def count_param_split(model: nn.Module):
    """Return (total_trainable_params, post_bottleneck_trainable_params).

    Required by the result JSON (see PARAM_MATCH_SPEC.md sec 3).
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    post = sum(p.numel() for p in model.post_bottleneck_parameters() if p.requires_grad)
    return total, post
