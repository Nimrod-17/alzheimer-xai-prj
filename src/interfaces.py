from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class DataLoaderInterface(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        '''
        Loads and returns a DataFrame containing the dataset.
        '''

class PreprocessorInterface(ABC):
    @abstractmethod
    def preprocess(self, file_path: str) -> np.ndarray:
        '''
        Loads and preprocesses an image, returns a numpy array.
        '''

class TrainerInterface(ABC):
    @abstractmethod
    def train_epoch(self, loader) -> float:
        '''
        Runs one epoch of training and returns the average loss.
        '''

    @abstractmethod
    def validate_epoch(self, loader) -> float:
        '''
        Runs one epoch of validation and returns the average loss.
        '''

class ModelSaverInterface(ABC):
    @abstractmethod
    def save(self, model, metric: float) -> None:
        '''
        Saves the model if the given metric is approved.
        '''

