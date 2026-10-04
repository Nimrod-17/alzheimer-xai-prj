import torch
from torch.utils.data import Dataset
import pandas as pd
from src.interfaces import PreprocessorInterface, AugmenterInterface

class MRIDataset(Dataset):
    """
    Custom PyTorch dataset for 3D MRI Alzheimer's classification.
    Depends on PreprocessorInterface (and optionally AugmenterInterface), not on any concrete implementation.
    """
    def __init__(self, dataframe: pd.DataFrame, preprocessor: PreprocessorInterface,
                 augmenter: AugmenterInterface | None = None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        file_path = self.dataframe.loc[idx, 'Path']
        label = self.dataframe.loc[idx, 'Label']

        # Delegate preprocessing (and optional augmentation) to the injected strategies
        img_array = self.preprocessor.preprocess(file_path)
        if self.augmenter is not None:
            img_array = self.augmenter.augment(img_array)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor
