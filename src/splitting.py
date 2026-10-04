import os

import numpy as np
import pandas as pd

from src.interfaces import SplitterInterface


class PersistentSubjectSplitter(SplitterInterface):
    '''
    Assign whole subjects to train/val/test, stratified by subject label, and remember the assignment.

    Every scan of a subject lands in the same split, so the model can never be evaluated on a brain
    it has already seen. Assignments are stored on disk: when the dataset grows (more downloads),
    existing subjects keep their split and only new subjects are distributed.
    '''
    SPLITS = ('train', 'val', 'test')

    def __init__(self, registry_path: str, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 42):
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError(f"Split ratios must sum to 1, got {ratios}")
        self.registry_path = registry_path
        self.ratios = ratios
        self.seed = seed

    def split(self, df: pd.DataFrame) -> pd.DataFrame:
        registry = self._load_registry()

        # A subject is positive if any of its scans is positive
        subject_labels = df.groupby('Subject')['Label'].max()
        new_subjects = subject_labels[~subject_labels.index.isin(registry.index)]

        if len(new_subjects) > 0:
            registry = pd.concat([registry, self._assign(new_subjects)])
            self._save_registry(registry)
            print(f"Assigned {len(new_subjects)} new subjects to splits (registry: {self.registry_path}).")

        df = df.copy()
        df['Split'] = df['Subject'].map(registry)
        self._report(df)
        return df

    def _assign(self, subject_labels: pd.Series) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        assignment = {}

        for _, group in subject_labels.groupby(subject_labels):
            subjects = group.index.to_numpy()
            rng.shuffle(subjects)

            boundaries = np.round(np.cumsum(self.ratios) * len(subjects)).astype(int)
            for split_name, chunk in zip(self.SPLITS, np.split(subjects, boundaries[:-1])):
                assignment.update({subject: split_name for subject in chunk})

        return pd.Series(assignment, name='Split')

    def _load_registry(self) -> pd.Series:
        if not os.path.exists(self.registry_path):
            return pd.Series(dtype=str, name='Split')
        return pd.read_csv(self.registry_path, index_col='Subject')['Split']

    def _save_registry(self, registry: pd.Series) -> None:
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        registry.rename_axis('Subject').sort_index().to_csv(self.registry_path)

    @staticmethod
    def _report(df: pd.DataFrame) -> None:
        for split_name, part in df.groupby('Split', sort=False):
            subjects = part.groupby('Subject')['Label'].max()
            print(f"  {split_name:<5} | {len(part):>4} scans | {len(subjects):>3} subjects "
                  f"(label 0: {(subjects == 0).sum()}, label 1: {(subjects == 1).sum()})")
