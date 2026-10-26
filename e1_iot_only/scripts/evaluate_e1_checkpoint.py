import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from train_e1_iot_mlp import E1MLP
from train_e1_iot_mlp import FEATURE_DIR
from train_e1_iot_mlp import META_COLS
from train_e1_iot_mlp import RESULTS_DIR
from train_e1_iot_mlp import SPLIT_DIR
from train_e1_iot_mlp import make_loader
from train_e1_iot_mlp import metrics_dict
from train_e1_iot_mlp import predict
from train_e1_iot_mlp import select_by_split


def load_csv(path):
    return pd.read_csv(path, dtype={"image": str, "label": str})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    split_dir = Path(args.split_dir)
    results_dir = Path(args.results_dir)

    metrics_path = results_dir / (args.run_name + "_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    all_train_features = load_csv(feature_dir / "train_iot_features.csv")
    all_test_features = load_csv(feature_dir / "test_iot_features.csv")

    train_split = load_csv(split_dir / "train.csv")
    val_split = load_csv(split_dir / "val.csv")
    test_split = load_csv(split_dir / "test.csv")

    train_df = select_by_split(all_train_features, train_split, "train")
    val_df = select_by_split(all_train_features, val_split, "val")
    test_df = select_by_split(all_test_features, test_split, "test")

    feature_cols = [c for c in train_df.columns if c not in META_COLS]

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_df[feature_cols])
    x_val = scaler.transform(val_df[feature_cols])
    x_test = scaler.transform(test_df[feature_cols])

    y_train = label_encoder.transform(train_df["label"])
    y_val = label_encoder.transform(val_df["label"])
    y_test = label_encoder.transform(test_df["label"])

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    val_loader = make_loader(x_val, y_val, args.batch_size, False, 0)
    test_loader = make_loader(x_test, y_test, args.batch_size, False, 0)

    model = E1MLP(
        input_dim=len(feature_cols),
        num_classes=len(label_encoder.classes_),
        hidden1=saved_metrics["hidden1"],
        hidden2=saved_metrics["hidden2"],
        dropout1=saved_metrics["dropout1"],
        dropout2=saved_metrics["dropout2"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    class_counts = torch.bincount(torch.tensor(y_train), minlength=len(label_encoder.classes_)).float()
    class_weights = len(y_train) / (len(label_encoder.classes_) * torch.clamp(class_counts, min=1.0))
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    val_output = predict(model, val_loader, criterion, device)
    test_output = predict(model, test_loader, criterion, device)

    classes = label_encoder.classes_.tolist()
    saved_metrics["val"] = metrics_dict(val_output, classes)
    saved_metrics["test"] = metrics_dict(test_output, classes)
    saved_metrics["metrics_updated_with_roc_auc"] = True

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(saved_metrics, f, indent=2)

    print("updated", metrics_path)
    print("val macro roc-auc", saved_metrics["val"]["macro_roc_auc_ovr"])
    print("test macro roc-auc", saved_metrics["test"]["macro_roc_auc_ovr"])


if __name__ == "__main__":
    main()
