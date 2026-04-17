import torch
from torch.utils.data import Dataset
import numpy as np
from src.preprocessing import load_and_preprocess_image

class MRIDataset(Dataset):
    """
    Custom PyToarch dataset for 3D MRI Alzheimer classification.
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)
    
    def __len__(self):
        return len(self.dataframe) # Return the total number of patients
    
    def __getitem__(self,idx):
        file_path = self.dataframe.loc[idx, 'Path']
        label = self.dataframe.loc[idx, 'Label']

        img_array = load_and_preprocess_image(file_path)

        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0) # Add channel dimension
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor