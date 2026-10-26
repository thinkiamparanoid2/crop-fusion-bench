"""Architecture contracts for the E1 encoder and the parameter-matched E2/E3 fusion heads."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "e3_quantum_fusion"))

from common.models import E1MLP  # noqa: E402
from models import FUSED_DIM, NUM_CLASSES, E2MatchedHead, E3HybridHead, count_param_split  # noqa: E402


def test_e1_embedding_is_the_64_dim_vector_fusion_consumes():
    model = E1MLP(input_dim=54, num_classes=NUM_CLASSES).eval()
    x = torch.randn(5, 54)
    assert model(x).shape == (5, NUM_CLASSES)
    assert model.embedding(x).shape == (5, 64)


def test_e2_matched_and_e3_vqc_heads_are_parameter_matched():
    e2, e3 = E2MatchedHead(), E3HybridHead()
    assert count_param_split(e2) == count_param_split(e3) == (173_073, 16)


def test_e3_forward_shape_bounds_and_gradient_reach_the_circuit():
    torch.manual_seed(0)
    model = E3HybridHead()
    x = torch.randn(6, FUSED_DIM)
    z = model.vqc(torch.pi * torch.tanh(model.projection(x)))
    assert z.shape == (6, 4)
    assert torch.all(z.abs() <= 1 + 1e-6)  # Pauli-Z expectations
    logits = model(x)
    assert logits.shape == (6, NUM_CLASSES)
    logits.sum().backward()
    grads = [p.grad for p in model.post_bottleneck_parameters()]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)
