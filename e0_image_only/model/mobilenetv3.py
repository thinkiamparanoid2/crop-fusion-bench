"""MobileNetV3-Large model for transfer learning with custom classifier head.

This module builds a MobileNetV3-Large model with frozen feature extractor
and a custom classifier head for 25 classes.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Respect the shared project training environment when it is already set.
# Otherwise keep pretrained-weight downloads inside the workspace cache.
TORCH_CACHE = Path(os.environ.get("TORCH_HOME", PROJECT_ROOT / ".cache" / "torch"))
TORCH_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(TORCH_CACHE)

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v3_large

from config import REPO_ROOT

NUM_CLASSES = 25


class MobileNetV3Classifier:
    """Builder for MobileNetV3-Large with frozen weights and custom classifier."""

    def __init__(self, num_classes: int = NUM_CLASSES, device: str = "cuda"):
        self.num_classes = num_classes
        self.device = device
        self.model = None
        self.trainable_param_count = 0

    def build(self) -> nn.Module:
        """Build the model with frozen feature extractor and custom classifier head."""
        print(f"Using torch cache: {os.environ['TORCH_HOME']}")
        print("Loading MobileNetV3-Large with pretrained ImageNet weights...")
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        print("Freezing all feature extractor weights...")
        for param in self.model.features.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[0].in_features

        print(f"Swapping classifier head for {self.num_classes} custom classes...")
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, self.num_classes),
        )

        self.model = self.model.to(self.device)
        self._count_trainable_params()

        return self.model

    def _count_trainable_params(self) -> None:
        """Count and log trainable parameters in the custom classifier head."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        self.trainable_param_count = trainable_params

        print("\n" + "=" * 70)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 70)
        print(f"Total Parameters:      {total_params:,}")
        print(f"Trainable Parameters:  {trainable_params:,}")
        print(f"Frozen Parameters:     {frozen_params:,}")
        print(f"Trainable % of Total:  {(trainable_params / total_params * 100):.2f}%")
        print("=" * 70 + "\n")

        log_path = REPO_ROOT / "model" / "trainable_params.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("MODEL PARAMETER SUMMARY\n")
            f.write("=" * 70 + "\n")
            f.write(f"Total Parameters:      {total_params:,}\n")
            f.write(f"Trainable Parameters:  {trainable_params:,}\n")
            f.write(f"Frozen Parameters:     {frozen_params:,}\n")
            f.write(f"Trainable % of Total:  {(trainable_params / total_params * 100):.2f}%\n")
            f.write("=" * 70 + "\n")

        print(f"Parameter summary saved to: {log_path}")

    def get_model(self) -> nn.Module:
        """Get the built model."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")
        return self.model

    def get_trainable_param_count(self) -> int:
        """Get the count of trainable parameters."""
        return self.trainable_param_count


def main() -> None:
    """Test the model builder."""
    print("Testing MobileNetV3Classifier...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    builder = MobileNetV3Classifier(num_classes=NUM_CLASSES, device=device)
    model = builder.build()

    print("Testing forward pass with dummy batch...")
    dummy_input = torch.randn(1, 3, 224, 224).to(builder.device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (1, {NUM_CLASSES})")

    if output.shape == (1, NUM_CLASSES):
        print("\nModel test passed.")
    else:
        print("\nModel test failed.")


if __name__ == "__main__":
    main()
