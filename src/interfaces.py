from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from src.brain_volume import BrainVolume

class DataLoaderInterface(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        '''
        Loads and returns a DataFrame containing the dataset.
        '''

class LabelingStrategyInterface(ABC):
    @abstractmethod
    def label(self, scans: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
        '''
        Adds a 'Label' column to the matched scans and drops rows that do not belong to any class.
        The full clinical history (visits) is available for strategies that look at follow-up.
        '''

class ScanSelectorInterface(ABC):
    @abstractmethod
    def select(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Chooses which labelled scans are kept in the final dataset.
        '''

class VolumeStepInterface(ABC):
    @abstractmethod
    def apply(self, volume: BrainVolume) -> BrainVolume:
        '''
        Applies one preprocessing step and returns the transformed volume.
        '''

class QCMetricInterface(ABC):
    @abstractmethod
    def compute(self, image: np.ndarray, mask: np.ndarray, affine_matrix: np.ndarray) -> dict[str, float]:
        '''
        Computes one or more quality measures for a preprocessed volume in MNI space.
        affine_matrix is the 3x3 linear part of the MNI -> native transform.
        '''

class AttributionMethodInterface(ABC):
    name: str

    @abstractmethod
    def attribute(self, model, volume, target: int) -> np.ndarray:
        '''
        Explains model(volume)[target] for a single input of shape (1, 1, D, H, W).
        Returns a (D, H, W) attribution map on the network input grid; positive = evidence for target.
        '''

class SplitterInterface(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Returns the dataframe with a 'Split' column ('train', 'val' or 'test').
        '''

class AugmenterInterface(ABC):
    @abstractmethod
    def augment(self, volume: np.ndarray) -> np.ndarray:
        '''
        Returns a randomly perturbed copy of a preprocessed volume.
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
    def evaluate(self, loader) -> tuple[float, np.ndarray, np.ndarray]:
        '''
        Runs inference on a loader and returns (average loss, true labels, probabilities of class 1).
        '''

class ModelSaverInterface(ABC):
    @abstractmethod
    def save(self, model, metric: float, extra: dict | None = None) -> bool:
        '''
        Saves the model (and optional metadata) if the given metric is approved.
        Returns True when a new checkpoint was written.
        '''

