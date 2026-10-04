import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from src.interfaces import PreprocessorInterface


class CropResizeLoader(PreprocessorInterface):
    '''
    Loader for volumes already preprocessed into MNI space (skull-stripped, bias-corrected, z-scored).
    Crops a fixed box around the brain and resamples it to the network input shape.
    Both operations are the same for every subject, so voxel positions stay comparable across the dataset.
    '''
    # Brain box in the 1 mm MNI152 grid (197 x 233 x 189), centred on the template brain with a margin
    DEFAULT_CROP = ((18, 178), (17, 217), (0, 160))

    def __init__(self, target_shape: tuple[int, int, int] = (128, 160, 128),
                 crop: tuple[tuple[int, int], ...] = DEFAULT_CROP):
        self.target_shape = target_shape
        self.crop = tuple(slice(start, stop) for start, stop in crop)

    def preprocess(self, file_path: str) -> np.ndarray:
        volume = np.asarray(nib.load(file_path).get_fdata(dtype=np.float32))[self.crop]

        if volume.shape == self.target_shape:
            return volume

        tensor = torch.from_numpy(np.ascontiguousarray(volume))[None, None]
        resized = F.interpolate(tensor, size=self.target_shape, mode='trilinear', align_corners=False)
        return resized[0, 0].numpy()
