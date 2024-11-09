import joblib
import torch
from sklearn.preprocessing import StandardScaler


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
