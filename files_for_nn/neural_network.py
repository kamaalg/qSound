from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np


def train_test_model(data_path, batch_size=2, num_epochs=100, hidden_dim=16, lr=0.001, test_size=0.2):
    # Load the data
    data = pd.read_csv(data_path, header=None)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Target variable

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define a simple neural network model
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = SimpleNN(input_size, hidden_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the scaler and model for future use
    #torch.save(model.state_dict(), 'trained_model3.pth')
    #np.save('scaler_mean3.npy', scaler.mean_)
    #np.save('scaler_scale3.npy', scaler.scale_)

    print("Model training complete.")
    return model, scaler


def test_model(new_data, scaler, model):
    # Scale the new data using the saved scaler
    new_data_scaled = (new_data - scaler.mean_) / scaler.scale_
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).view(1, -1)

    # Load the trained model
    model.eval()
    with torch.no_grad():
        prediction = model(new_data_tensor)

    print("Prediction for new data:", prediction.item())


# Example usage:
model, scaler = train_test_model('music_features.csv', batch_size=4, num_epochs=50, hidden_dim=32, lr=0.001,
                                 test_size=0.2)

# Test with the new numpy array
new_data = np.array([
    8.75000000e+02, 3.31905842e-01, 3.48522297e+03, -1.65006805e+02,
    8.30743027e+01, -2.60957432e+01, 8.76836166e+01, -4.93207130e+01,
    4.40938339e+01, -2.96863461e+01, 3.85098845e-01, -1.57739773e+01,
    2.87773466e+00, 3.62649584e+00, 1.44673100e+01, -1.09693012e+01,
    6.86058521e+00, -3.82035017e+00, -6.30810559e-01, -3.42632031e+00,
    -3.62196493e+00, -1.35855789e+01, -6.26379824e+00, 0.00000000e+00
])

test_model(new_data, scaler, model)