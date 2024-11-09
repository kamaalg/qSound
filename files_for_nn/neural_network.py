#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
data = pd.read_csv('audio_features_with_arousal.csv')

X = data.drop(columns=['arousal']).values
y = data['arousal'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2= nn.Linear(64, 32)
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


#Initialize the model
input_size = X_train.shape[1]
print(input_size)
model = FeedforwardNN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
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

model.eval()
test_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()

# Print average test loss
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")


# In[5]:


np.save("X_train.npy", X_train)


# In[6]:


import torch
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set the model to evaluation mode
model.eval()

# Lists to hold predictions and actual values for metrics
all_predictions = []
all_actuals = []

# Measure the time taken for a single prediction
num_trials = 100  # Number of times to repeat the single prediction to get an average
start_time = time.time()

# Run the model without gradient calculation
with torch.no_grad():
    # Loop over test data
    for X_batch, y_batch in test_loader:
        # Measure time taken for single feature prediction
        for _ in range(num_trials):
            # Take only the first feature vector from the batch for a single prediction test
            single_feature_vector = X_batch[0].unsqueeze(0)  # Shape (1, num_features)

            # Start timing for the prediction
            pred_start = time.time()
            single_prediction = model(single_feature_vector)
            pred_end = time.time()

            # Output the prediction time for this single prediction
            print(f"Single Prediction Time: {pred_end - pred_start:.6f} seconds")

            # Add to overall list of predictions and actuals for metric calculations
            all_predictions.extend(model(X_batch).cpu().numpy().flatten())  
            all_actuals.extend(y_batch.cpu().numpy().flatten()) 

    # Calculate elapsed time for the num_trials predictions
    avg_time = (time.time() - start_time) / num_trials
    print(f"\nAverage Prediction Time for a Single Feature Vector: {avg_time:.6f} seconds")

# Calculate metrics
mse = mean_squared_error(all_actuals, all_predictions)
mae = mean_absolute_error(all_actuals, all_predictions)
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")

average_diff = sum(abs(pred - actual) for pred, actual in zip(all_predictions, all_actuals)) / len(all_predictions)
max_diff = max(abs(pred - actual) for pred, actual in zip(all_predictions, all_actuals))

print("Average Mistake:", average_diff)
print("Max Mistake:", max_diff)


# In[7]:


model_path = "entire_model.pth"  # Define the path for saving the model

# Save the model's state dictionary
torch.save(model, model_path)
print(f"Entire model saved")


# In[8]:


# import librosa
# import numpy as np
# import torch
# from sklearn.preprocessing import StandardScaler
# 
# def extract_features(file_path, duration=1):
#     y, sr = librosa.load(file_path, sr=44100, duration=duration)
#     tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
#     rms = np.mean(librosa.feature.rms(y=y))
#     spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
#     zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
#     mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
#     features = np.hstack([tempo, rms, spectral_centroid, zero_crossing_rate, mfcc])
#     return features
# 
# # Load and scale the features
# file_path = "path/to/your/new_audio_file.wav"
# new_features = extract_features(file_path).reshape(1, -1)
# 
# scaler = StandardScaler()
# scaler.fit(X_train)
# new_features = scaler.transform(new_features)
# 
# # Convert to a PyTorch tensor
# new_features_tensor = torch.tensor(new_features, dtype=torch.float32)
# 
# # Load model and make a prediction
# model.eval()
# with torch.no_grad():
#     prediction = model(new_features_tensor)
# print(f"Predicted Arousal Value: {prediction.item()}")


# In[9]:


scalar = StandardScaler()

scalar.fit(X_train)

joblib.dump(scaler, "scaler.pkl")


# In[10]:


def final_nn(features):
    entire_scalar = joblib.load('scaler.pkl')
    features = features.reshape(1,-1)
    new_features_model = entire_scalar.transform(features)
    new_features_tensor_model = torch.tensor(new_features_model, dtype=torch.float32)
    entire_model = torch.load("entire_model.pth")
    entire_model.eval()

    while torch.no_grad():
        model_prediction = entire_model(new_features_tensor_model)
        print("Predicted Intensity value is, pred", model_prediction)


# In[10]:




