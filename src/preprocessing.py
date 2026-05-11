import numpy as np
import nibabel as nib
from src.interfaces import PreprocessorInterface

class MinMaxNormalizer(PreprocessorInterface):
    '''
    SRP: single responsibility — apply min-max normalization to a 3D MRI image.
    OCP: to add z-score or other strategies, create a new class without touching this one.
    '''
    def preprocess(self, file_path: str) -> np.ndarray:
        img = nib.load(file_path)
        img_data = np.squeeze(img.get_fdata())

        min_val = np.min(img_data)
        max_val = np.max(img_data)

        # Avoid division by zero for completely blank images
        if max_val - min_val > 0:
            return (img_data - min_val) / (max_val - min_val)
        
        return img_data

class ZScoreNormalizer(PreprocessorInterface):
    '''
    Alternative preprocessor using z-score standardization.
    Drops in as a replacement anywhere PreprocessorInterface is expected.
    '''
    def preprocess(self, file_path: str) -> np.ndarray:
        img = nib.load(file_path)
        img_data = np.squeeze(img.get_fdata())
        std = np.std(img_data)
        if std > 0:
            return (img_data - np.mean(img_data)) / std
        
        return img_data