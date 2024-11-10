import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)       # Increased units
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)                # Final output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        # Reduced dropout

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

          # Apply tanh to limit output to [-1, 1]

        return x



import numpy as np
import torch
import pandas as pd


def sigmoid_normalize(value, scaling_factor=0.01):
    # Apply sigmoid and scale to [-1, 1]
    sigmoid_value = 1 / (1 + np.exp(-value * scaling_factor))  # Scaling factor adjusts spread
    normalized_value = (sigmoid_value * 2) - 1
    return normalized_value

def normalize_individual_values(tensor, data_min, data_max, min_value=-1, max_value=1):
    # Apply min-max scaling to each value in the tensor
    normalized_tensor = (tensor - data_min) / (data_max - data_min)  # Scale each value to [0, 1]
    normalized_tensor = normalized_tensor * (
                max_value - min_value) + min_value  # Scale each value to [min_value, max_value]
    return normalized_tensor



def final_nn(features):
    scalar = StandardScaler()
    data = pd.read_csv('/Users/brody/PycharmProjects/q-Sound/qSound/files_for_nn/audio_features_with_arousal.csv')
    X = data.drop(columns=['arousal']).values
    y = data['arousal'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scalar.fit(X_train)
    # Define means and standard deviations for normalization

    normalized_feature = features.reshape(1, -1)
    # feature_means = np.array([
    #     1.20614275e+02, 1.08991174e-01, 2.02186075e+03, 4.52432108e-02,
    #     -2.45413664e+02, 1.64192899e+02, -1.47724596e+01, 3.33616434e+01,
    #     -9.39535519e-01, 1.74323718e+01, -6.21402045e+00, 7.97688733e+00,
    #     -3.30131782e+00, 2.30141983e+00, -3.08561731e+00, 3.06089681e-01,
    #     -1.59249887e+00, -1.67337213e+00, -9.11131115e-01, -2.93700843e+00,
    #     -6.54796014e-01, -3.32906958e+00, 5.07976066e-01, -3.39474129e+00
    # ])
    # feature_stds = np.array([
    #     3.17174287e+01, 6.63909309e-02, 9.01628955e+02, 2.34573042e-02,
    #     9.37549767e+01, 3.40698684e+01, 3.42174005e+01, 2.56557258e+01,
    #     1.61135376e+01, 1.45377345e+01, 1.14398191e+01, 1.16079697e+01,
    #     8.37968536e+00, 9.44078965e+00, 7.37284219e+00, 7.29980293e+00,
    #     5.99741298e+00, 6.08417760e+00, 5.53784999e+00, 4.76320286e+00,
    #     5.13831387e+00, 4.55852179e+00, 5.06443742e+00, 4.59918779e+00
    # ])
    # Normalize the new data using training set mean and std
    # normalized_data = (normalized_feature - feature_means) / feature_stds
    normalized_feature = scalar.transform(normalized_feature)
    # Normalize the input feature vector
    # normalized_feature = (features - feature_means) / feature_stds
    new_features_tensor_model = torch.tensor(normalized_feature, dtype=torch.float32)

    # Load the model and initialize it with the saved state
    input_size = new_features_tensor_model.shape[1]
    model = FeedforwardNN(input_size)
    model.load_state_dict(torch.load("/Users/brody/PycharmProjects/q-Sound/qSound/files_for_nn/model_state.pth"))
    model.eval()

    # Make prediction
    with torch.no_grad():
        model_prediction = model(new_features_tensor_model)

    # Apply softmax to the mean tensor
    return sigmoid_normalize(model_prediction[0].item())



# Min-max normalization to [0, 1]
# final_nn(features=np.array([
#     1.63043478e+02, 4.85700741e-02, 3.16967797e+03, -2.90725250e+02,
#     1.28058182e+02, 3.10450954e+01, 4.83565407e+01, -4.77100601e+01,
#     2.52849617e+01, -4.36648560e+01, 1.72412243e+01, -1.44654036e+01,
#     1.24386721e+01, -1.29378157e+01, -9.09165764e+00, -5.12640858e+00,
#     -7.03804636e+00, -8.69009972e+00, -8.59643936e+00, -1.11997232e+01,
#     -3.28026581e+00, 2.18192097e-02, -9.62158203e+00, 0.00000000e+00
# ]))
base_array = np.array([
    1.08173077e+01, 7.12440237e-02, 2.84565367e+03, -2.00334396e+02,
    1.51824493e+02, -2.55831089e+01, 5.41809807e+01, -2.31092834e+01,
    3.03525581e+01, -1.58262939e+01, 4.35570860e+00, -8.10534668e+00,
    1.11825199e+01, 8.30592155e+00, 1.46151018e+01, 1.02710285e+01,
    7.18000937e+00, 8.05258369e+00, -3.22280574e+00, 1.27353144e+00,
    -3.06853342e+00, 5.75506783e+00, -8.22137713e-01, 0.00000000e+00
])

# Generate 100 arrays with small variations
arrays = [base_array + np.random.normal(0, 10, base_array.shape) for _ in range(100)]

# Print the first few arrays to verify
for i, array in enumerate(arrays):
    print(final_nn(array))