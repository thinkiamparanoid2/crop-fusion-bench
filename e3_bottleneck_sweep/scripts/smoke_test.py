"""Smoke test bottleneck-8 E2-matched / E3 VQC shapes and parameter counts."""

import sys
from pathlib import Path

import torch

SWEEP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SWEEP_ROOT.parent
for _p in (PROJECT_ROOT, SWEEP_ROOT / "scripts", PROJECT_ROOT / "e2_multimodal_fusion" / "scripts", PROJECT_ROOT / "e3_quantum_fusion"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_bottleneck import E2MatchedK, E3VQCK, count_param_split  # noqa: E402
from train_e2_fusion import load_split  # noqa: E402


def main():
    feature_dir = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
    x, y, meta = load_split(feature_dir, "val")
    batch = torch.tensor(x[:8], dtype=torch.float32)

    expected_total = 173737
    expected_post = 64
    for name, model in [
        ("e2_matched_8", E2MatchedK(8)),
        ("e3_8_vqc", E3VQCK(8, 4, sim="lightning.qubit")),
    ]:
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        total, post = count_param_split(model)
        assert logits.shape == (8, 25), (name, logits.shape)
        assert total == expected_total, (name, total, expected_total)
        assert post == expected_post, (name, post, expected_post)
        print(f"[ok] {name}: logits={tuple(logits.shape)} total={total} post={post}")

    print("bottleneck-8 smoke test passed")


if __name__ == "__main__":
    main()

