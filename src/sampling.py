from collections.abc import Iterator

import pandas as pd
import torch
from torch.utils.data import Sampler


class SubjectBalancedSampler(Sampler[int]):
    '''
    Each epoch draws exactly one random scan per subject, in random order.

    Subjects with many sessions do not dominate training (no incentive to memorize individual brains),
    while across epochs all of their scans are still seen, acting as natural data augmentation.
    '''
    def __init__(self, subjects: pd.Series, seed: int = 42):
        # Dataset row positions grouped by subject
        positions = pd.Series(range(len(subjects)), index=subjects.to_numpy())
        self.groups = [torch.tensor(group.to_numpy()) for _, group in positions.groupby(level=0)]
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[int]:
        picks = [group[torch.randint(len(group), (1,), generator=self.generator)].item() for group in self.groups]
        order = torch.randperm(len(picks), generator=self.generator)
        return iter([picks[i] for i in order])
