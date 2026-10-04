'''
Composition helpers shared by the entry-point scripts (training, evaluation, XAI),
so that every script builds data and model in exactly the same way.
'''
from src.augmentation import RandomShiftIntensityAugmenter
from src.data_module import OASIS3DataModule
from src.model import ResNet3DClassifier
from src.mri_pipeline import ProcessedVolumeWriter
from src.oasis3_parser import ClosestVisitPerSubjectSelector
from src.preprocessing import CropResizeLoader
from src.splitting import PersistentSubjectSplitter

# --- Shared configuration ---
LABELS_CSV = 'data/processed/oasis3_ad_risk.csv'
PREPROCESSED_DIR = 'data/preprocessed'
SPLIT_REGISTRY = 'data/processed/subject_splits.csv'
INPUT_SHAPE = (128, 160, 128)
RANDOM_SEED = 42


def build_data_module(batch_size: int = 8, num_workers: int = 4, augment: bool = True) -> OASIS3DataModule:
    return OASIS3DataModule(
        labels_csv=LABELS_CSV,
        volume_store=ProcessedVolumeWriter(PREPROCESSED_DIR),
        splitter=PersistentSubjectSplitter(SPLIT_REGISTRY, seed=RANDOM_SEED),
        eval_selector=ClosestVisitPerSubjectSelector(),
        preprocessor=CropResizeLoader(target_shape=INPUT_SHAPE),
        train_augmenter=RandomShiftIntensityAugmenter() if augment else None,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=RANDOM_SEED,
    )


def build_model() -> ResNet3DClassifier:
    return ResNet3DClassifier()
