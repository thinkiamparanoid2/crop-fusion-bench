"""E3 shape + parameter-count smoke test (STUB).

Implements the Test Plan in docs/plans/E3_STRUCTURE_AND_IMPLEMENTATION_PLAN.md:
validates the embedding interface, the forward output shape, and the Spec B
parameter accounting on the REAL cached embeddings (no training).

Run (once implemented):
    python e3_quantum_fusion/scripts/smoke_test.py
"""

import sys
from pathlib import Path

E3_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = E3_ROOT.parent
E2_SCRIPTS = PROJECT_ROOT / "e2_multimodal_fusion" / "scripts"
for _p in (PROJECT_ROOT, E3_ROOT, E2_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"

# Expected (locked) values to assert:
EXPECT_FUSED_DIM = 1344
EXPECT_NUM_CLASSES = 25
EXPECT_TOTAL_PARAMS = 173073      # E2-matched and E3-4
EXPECT_POST_BOTTLENECK = 16       # matched decision sub-network / VQC


def main():
    import torch

    from train_e2_fusion import load_split
    from models import BOTTLENECK, E2MatchedHead, E3HybridHead, count_param_split

    x, y, meta = load_split(FEATURE_DIR, "val")
    assert x.shape[1] == EXPECT_FUSED_DIM, f"fused input dim {x.shape[1]} != {EXPECT_FUSED_DIM}"
    print(f"[ok] val embeddings: x={x.shape}, y={y.shape}, fused dim={x.shape[1]}")

    batch = torch.tensor(x[:8], dtype=torch.float32)

    for name, Head in (("E2-matched", E2MatchedHead), ("E3-4", E3HybridHead)):
        model = Head()
        model.eval()

        # Confirm the decision sub-network receives exactly [B, 4].
        bottleneck = model.projection(batch)
        assert bottleneck.shape == (8, BOTTLENECK), f"{name} bottleneck {tuple(bottleneck.shape)} != (8, {BOTTLENECK})"

        with torch.no_grad():
            logits = model(batch)
        assert logits.shape == (8, EXPECT_NUM_CLASSES), f"{name} logits {tuple(logits.shape)} != (8, {EXPECT_NUM_CLASSES})"

        total, post = count_param_split(model)
        assert total == EXPECT_TOTAL_PARAMS, f"{name} total params {total} != {EXPECT_TOTAL_PARAMS}"
        assert post == EXPECT_POST_BOTTLENECK, f"{name} post-bottleneck params {post} != {EXPECT_POST_BOTTLENECK}"

        print(
            f"[ok] {name}: bottleneck={tuple(bottleneck.shape)} logits={tuple(logits.shape)} "
            f"total_params={total} post_bottleneck_params={post}"
        )

    print("\nSmoke test passed: shapes and Spec B parameter accounting verified.")


if __name__ == "__main__":
    main()
