import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_and_validate(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, save_path='models/best_model.pth'):
    """
    Trains the 3D CNN model using the provided data loader and evaluates it on a validation set.
    """

    # Hardware selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f"Using device: {device}")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    print("--- Starting Training & Validation Loop ---")

    for epoch in range(num_epochs):

        # Training
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / max(1, len(val_loader))

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Saving
        if avg_val_loss < best_val_loss:
            print(f"New best model found! (Val Loss decreased from {best_val_loss:.4f} to {avg_val_loss:.4f}). Saving...")
            best_val_loss = avg_val_loss

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print("Training complete")

    return model