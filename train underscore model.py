
import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    EPOCHS,
    LEARNING_RATE,
    DEVICE,
    MODEL_DIR
)

from models import create_model
from data_preprocessing import main as load_preprocessed_data


def train_model():

    # Load preprocessed data
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler
    ) = load_preprocessed_data()

    # Convert NumPy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Move data to device
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_val = X_val.to(DEVICE)
    y_val = y_val.to(DEVICE)

    # Create model
    model = create_model()
    model = model.to(DEVICE)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Training loop
    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        predictions = model(X_train)

        loss = criterion(predictions, y_train)

        loss.backward()

        optimizer.step()

        # Validation
        model.eval()

        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, y_val)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}], "
                f"Training Loss: {loss.item():.6f}, "
                f"Validation Loss: {val_loss.item():.6f}"
            )

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        "mlp_model.pth"
    )

    torch.save(model.state_dict(), model_path)

    print("Training completed successfully!")
    print(f"Model saved at: {model_path}")

    return model


if __name__ == "__main__":
    train_model()

