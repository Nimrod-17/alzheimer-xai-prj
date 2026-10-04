import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from src.factories import build_data_module, build_model, RANDOM_SEED
from src.trainer import ModelTrainer, BestModelSaver, EarlyStopping, TrainingHistoryLogger, TrainingOrchestrator

# --- Configuration ---
SAVE_PATH = 'models/oasis3_resnet3d.pth'
HISTORY_PATH = 'runs/oasis3_resnet3d_history.csv'
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 150
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 30


def main():
    parser = argparse.ArgumentParser(description='Train the 3D CNN on OASIS-3.')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)

    # 1. Subject-level splits and loaders (only preprocessed scans are used)
    data = build_data_module(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    data.setup()

    # 2. Select compute device and initialise model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = build_model().to(device)

    # 3. Compose the training pipeline (all dependencies injected — nothing hardcoded)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=data.class_weights().to(device))

    trainer = ModelTrainer(model=model, optimizer=optimizer, criterion=criterion, device=device)
    orchestrator = TrainingOrchestrator(
        trainer=trainer,
        saver=BestModelSaver(save_path=SAVE_PATH, mode='max'),
        num_epochs=args.epochs,
        monitor='auc',
        scheduler=scheduler,
        early_stopping=EarlyStopping(patience=PATIENCE, mode='max'),
        history_logger=TrainingHistoryLogger(HISTORY_PATH),
    )

    # 4. Start training
    orchestrator.run(data.train_loader(), data.val_loader())


if __name__ == '__main__':
    main()
