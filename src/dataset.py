import torch
from torch.utils.data import Dataset
import pandas as pd
from src.interfaces import PreprocessorInterface

class MRIDataset(Dataset):
    """
    Custom PyTorch dataset for 3D MRI Alzheimer's classification.
    Depends on PreprocessorInterface, not on any concrete implementation.
    """
    def __init__(self, dataframe: pd.DataFrame, preprocessor: PreprocessorInterface):
        self.dataframe = dataframe.reset_index(drop=True)
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        file_path = self.dataframe.loc[idx, 'Path']
        label = self.dataframe.loc[idx, 'Label']

        # Delegate preprocessing to the injected strategy
        img_array = self.preprocessor.process(file_path)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor