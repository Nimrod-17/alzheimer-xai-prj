import json

import torch
import torch.nn as nn

from src.factories import build_data_module, build_model
from src.metrics import BinaryClassificationMetrics, BootstrapConfidenceInterval
from src.trainer import ModelTrainer

# --- Configuration ---
MODEL_PATH = 'models/oasis3_resnet3d.pth'
METADATA_PATH = 'models/oasis3_resnet3d.json'
BATCH_SIZE = 8


def main():
    print("--- Clinical evaluation on the held-out test subjects ---")

    # 1. Load the best checkpoint and the threshold chosen on validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    with open(METADATA_PATH, encoding='utf-8') as metadata_file:
        checkpoint_info = json.load(metadata_file)
    threshold = checkpoint_info['threshold']
    print(f"Checkpoint from epoch {checkpoint_info['epoch']} (val AUC {checkpoint_info['auc']:.3f}), "
          f"threshold {threshold:.3f}")

    # 2. Test subjects: one fixed scan each, never seen during training or model selection
    data = build_data_module(batch_size=BATCH_SIZE, augment=False)
    data.setup()

    trainer = ModelTrainer(model=model, optimizer=None, criterion=nn.CrossEntropyLoss(), device=device)
    _, labels, probabilities = trainer.evaluate(data.test_loader())

    # 3. Metrics with 95% bootstrap confidence intervals
    metrics = BinaryClassificationMetrics()
    results = metrics.compute(labels, probabilities, threshold)
    intervals = BootstrapConfidenceInterval(metrics).compute(labels, probabilities, threshold)

    # 4. Report
    print("=" * 52)
    print(f" TEST REPORT — {len(labels)} subjects ({int(labels.sum())} positive)")
    print("=" * 52)
    for name in ('auc', 'balanced_accuracy', 'sensitivity', 'specificity'):
        low, high = intervals[name]
        print(f"{name:<18}: {results[name]:.3f}  (95% CI {low:.3f}–{high:.3f})")
    print(f"{'f1':<18}: {results['f1']:.3f}")
    print("-" * 52)
    print("CONFUSION MATRIX:")
    print(f"  True Positive (TP) : {results['tp']:>3} | False Positive (FP): {results['fp']:>3}")
    print(f"  False Negative (FN): {results['fn']:>3} | True Negative (TN) : {results['tn']:>3}")
    print("=" * 52)


if __name__ == '__main__':
    main()
