import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.data_loader import ExcelClinicalParser, MRIFileResolver, MRIDataLoader
from src.preprocessing import MinMaxNormalizer
from src.dataset import MRIDataset
from src.model import Alzheimer3DCNN
from src.trainer import ModelTrainer, BestModelSaver, TrainingOrchestrator

# --- Configuration ---
EXCEL_PATH = 'data/Demographic and Clinical Data/oasis_cross-sectional.xlsx'
DATA_DIR = 'data/cleaned_data'
SAVE_PATH = 'models/best_model.pth'
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2

def main():
    # 1. Load and match clinical data with MRI file paths
    parser = ExcelClinicalParser(excel_path=EXCEL_PATH)
    resolver = MRIFileResolver(base_data_dir=DATA_DIR, search_pattern="{id}_stripped.nii.gz")
    data_loader = MRIDataLoader(parser=parser, resolver=resolver)
    df = data_loader.load_data()

    # 2. Build dataset and split into training / validation sets
    preprocessor = MinMaxNormalizer()
    dataset = MRIDataset(dataframe=df, preprocessor=preprocessor)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Select compute device and initialise model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = Alzheimer3DCNN().to(device)

    # 4. Compose the training pipeline (all dependencies injected — nothing hardcoded)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    trainer = ModelTrainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
    saver = BestModelSaver(save_path=SAVE_PATH)
    orchestrator = TrainingOrchestrator(trainer=trainer, saver=saver, num_epochs=NUM_EPOCHS)

    # 5. Start training
    orchestrator.run(train_loader, val_loader)


if __name__ == '__main__':
    main()