import torch
import os
from src.interfaces import TrainerInterface, ModelSaverInterface

class BestModelSaver(ModelSaverInterface):
    '''
    Persist the model whenever validation loss improves.
    '''
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.best_metric = float('inf')

    def save(self, model, metric: float) -> None:
        if metric < self.best_metric:
            print(f"  → New best! Loss {self.best_metric:.4f} → {metric:.4f}. Saving...")
            self.best_metric = metric
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)

class ModelTrainer(TrainerInterface):
    '''
    SRP: owns only the training and validation logic for a single epoch.
    DIP: criterion and optimizer are injected — nothing is hardcoded here.
    '''
    def __init__(self, model, optimizer, criterion, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def validate_epoch(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                loss = self.criterion(self.model(images), labels)
                total_loss += loss.item()

        return total_loss / max(1, len(loader))


class TrainingOrchestrator:
    '''
    SRP: coordinates the full epoch loop — unaware of internal training details.
    OCP: num_epochs, trainer, and saver are all replaceable without modifying this class.
    '''
    def __init__(self, trainer: TrainerInterface, saver: ModelSaverInterface, num_epochs: int = 30):
        self.trainer = trainer
        self.saver = saver
        self.num_epochs = num_epochs

    def run(self, train_loader, val_loader) -> None:
        print("--- Starting Training & Validation Loop ---")

        for epoch in range(self.num_epochs):
            train_loss = self.trainer.train_epoch(train_loader)
            val_loss = self.trainer.validate_epoch(val_loader)

            print(f"Epoch [{epoch+1}/{self.num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            self.saver.save(self.trainer.model, val_loss)

        print("Training complete.")
