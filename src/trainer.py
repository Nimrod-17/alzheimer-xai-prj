import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, num_epochs=5, learning_rate=0.001):
    """
    Trains the 3D CNN model using the provided data loader.
    """

    # Hardware selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print (f"Using device: {device}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("--- Starting Training ---")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"   Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    return model