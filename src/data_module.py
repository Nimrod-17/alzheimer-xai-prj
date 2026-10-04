import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import MRIDataset
from src.interfaces import AugmenterInterface, PreprocessorInterface, ScanSelectorInterface, SplitterInterface
from src.mri_pipeline import ProcessedVolumeWriter
from src.sampling import SubjectBalancedSampler


class OASIS3DataModule:
    '''
    Build the train/val/test DataLoaders from the labelled OASIS-3 metadata.

    - Splits are made per subject (via the injected splitter) on the full label table,
      so they do not change while preprocessing is still running.
    - Only scans whose preprocessed volume exists on disk are used.
    - Training draws one random scan per subject per epoch; validation and test use one fixed
      scan per subject (via the injected selector), so every subject counts exactly once in the metrics.
    '''
    def __init__(
        self,
        labels_csv: str,
        volume_store: ProcessedVolumeWriter,
        splitter: SplitterInterface,
        eval_selector: ScanSelectorInterface,
        preprocessor: PreprocessorInterface,
        train_augmenter: AugmenterInterface | None = None,
        batch_size: int = 4,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self.labels_csv = labels_csv
        self.volume_store = volume_store
        self.splitter = splitter
        self.eval_selector = eval_selector
        self.preprocessor = preprocessor
        self.train_augmenter = train_augmenter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.frames: dict[str, pd.DataFrame] = {}

    def setup(self) -> None:
        print("--- Building OASIS-3 data splits ---")
        df = self.splitter.split(pd.read_csv(self.labels_csv))

        df['Path'] = df['ID'].map(lambda scan_id: str(self.volume_store.image_path(scan_id)))
        available = df['ID'].map(self.volume_store.exists)
        print(f"Preprocessed volumes available: {available.sum()}/{len(df)} scans.")
        df = df[available]

        self.frames = {
            'train': df[df['Split'] == 'train'].reset_index(drop=True),
            'val': self.eval_selector.select(df[df['Split'] == 'val']),
            'test': self.eval_selector.select(df[df['Split'] == 'test']),
        }
        for split_name, frame in self.frames.items():
            counts = frame.groupby('Subject')['Label'].max().value_counts().sort_index().to_dict()
            print(f"  {split_name:<5} loader | {len(frame):>4} scans | {frame['Subject'].nunique():>3} subjects | subject labels {counts}")

    def class_weights(self) -> torch.Tensor:
        '''
        Inverse-frequency weights computed on training subjects (what the balanced sampler actually sees).
        '''
        subject_labels = self.frames['train'].groupby('Subject')['Label'].max()
        counts = subject_labels.value_counts().sort_index()
        weights = len(subject_labels) / (len(counts) * counts)
        return torch.tensor(weights.to_numpy(), dtype=torch.float32)

    def train_loader(self) -> DataLoader:
        frame = self.frames['train']
        return self._loader(
            MRIDataset(frame, self.preprocessor, self.train_augmenter),
            sampler=SubjectBalancedSampler(frame['Subject'], seed=self.seed),
        )

    def val_loader(self) -> DataLoader:
        return self._loader(MRIDataset(self.frames['val'], self.preprocessor))

    def test_loader(self) -> DataLoader:
        return self._loader(MRIDataset(self.frames['test'], self.preprocessor))

    def _loader(self, dataset: MRIDataset, sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
