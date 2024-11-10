import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def final_nn(features):
    # Load the saved scaler
    # data = pd.read_csv('files_for_nn/audio_features_with_arousal.csv')

    # X = data.drop(columns=['arousal']).values
    # y = data['arousal'].values
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    entire_scalar = joblib.load('files_for_nn/scaler.pkl')

    features = features.reshape(1, -1)
    new_features_model = entire_scalar.transform(np.load("files_for_nn/X_train.npy"))
    new_features_tensor_model = torch.tensor(new_features_model, dtype=torch.float32)

    # Load the model and initialize with the saved state
    input_size = new_features_tensor_model.shape[1]
    model = FeedforwardNN(input_size)
    model.load_state_dict(torch.load("files_for_nn/entire_model.pth"))
    model.eval()

    # Make prediction
    with torch.no_grad():
        model_prediction = model(new_features_tensor_model)
    print("Predicted Intensity value is:", model_prediction[0].item())
    return model_prediction[0].item()
