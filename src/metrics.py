import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class BinaryClassificationMetrics:
    '''
    Clinical metrics for a binary classifier, given true labels and predicted probabilities of class 1.
    '''
    def compute(self, labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
        predictions = (probabilities >= threshold).astype(int)

        tp = int(((predictions == 1) & (labels == 1)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())

        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0

        return {
            'auc': roc_auc_score(labels, probabilities) if len(np.unique(labels)) == 2 else float('nan'),
            'balanced_accuracy': (sensitivity + specificity) / 2,
            'accuracy': (tp + tn) / len(labels) if len(labels) else 0.0,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'threshold': threshold,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        }


class YoudenThresholdSelector:
    '''
    Pick the decision threshold maximizing Youden's J (sensitivity + specificity - 1).
    Must be fitted on validation data only, then reused unchanged on the test set.
    '''
    def select(self, labels: np.ndarray, probabilities: np.ndarray) -> float:
        if len(np.unique(labels)) < 2:
            return 0.5
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        best = np.argmax(tpr - fpr)
        return float(min(thresholds[best], 1.0))


class BootstrapConfidenceInterval:
    '''
    Percentile bootstrap confidence intervals, resampling subjects with replacement.
    Important with small test sets, where a single point estimate can be misleading.
    '''
    def __init__(self, metrics: BinaryClassificationMetrics, n_resamples: int = 2000,
                 confidence: float = 0.95, seed: int = 42):
        self.metrics = metrics
        self.n_resamples = n_resamples
        self.confidence = confidence
        self.seed = seed

    def compute(self, labels: np.ndarray, probabilities: np.ndarray, threshold: float,
                names: tuple[str, ...] = ('auc', 'balanced_accuracy', 'sensitivity', 'specificity')) -> dict:
        rng = np.random.default_rng(self.seed)
        samples = {name: [] for name in names}

        for _ in range(self.n_resamples):
            idx = rng.integers(0, len(labels), len(labels))
            if len(np.unique(labels[idx])) < 2:
                continue
            result = self.metrics.compute(labels[idx], probabilities[idx], threshold)
            for name in names:
                samples[name].append(result[name])

        alpha = (1 - self.confidence) / 2
        return {name: (float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha)))
                for name, values in samples.items()}
