import numpy as np
import torch

from src.interfaces import AugmenterInterface


class RandomShiftIntensityAugmenter(AugmenterInterface):
    '''
    Light augmentation suited to volumes already aligned to MNI space:
    a small random translation (residual registration error) and a random
    intensity scale/offset applied to brain voxels only (scanner variability).
    Uses torch's RNG so each DataLoader worker draws different values.
    '''
    def __init__(self, max_shift_voxels: int = 4, scale_range: float = 0.1, offset_range: float = 0.1):
        self.max_shift_voxels = max_shift_voxels
        self.scale_range = scale_range
        self.offset_range = offset_range

    def augment(self, volume: np.ndarray) -> np.ndarray:
        brain = volume != 0
        scale = 1.0 + self._uniform(self.scale_range)
        offset = self._uniform(self.offset_range)

        augmented = np.where(brain, volume * scale + offset, 0.0).astype(np.float32)
        return self._shift(augmented)

    def _shift(self, volume: np.ndarray) -> np.ndarray:
        '''Translate by an integer number of voxels per axis, filling the gap with background.'''
        shifts = torch.randint(-self.max_shift_voxels, self.max_shift_voxels + 1, (volume.ndim,)).tolist()
        shifted = np.zeros_like(volume)

        target, source = [], []
        for shift, size in zip(shifts, volume.shape):
            target.append(slice(max(shift, 0), size + min(shift, 0)))
            source.append(slice(max(-shift, 0), size + min(-shift, 0)))

        shifted[tuple(target)] = volume[tuple(source)]
        return shifted

    @staticmethod
    def _uniform(half_range: float) -> float:
        return (torch.rand(1).item() * 2 - 1) * half_range
