import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import psutil
 
 
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)
 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
 
 
# Step 1: Load and preprocess the data
df = pd.read_csv('Process_1 IDS-IoT-2024.csv')
 
# Drop the target variable and assign features to X
X = df.iloc[:, :-1].values
 
# Assign target variable to y
y = df['Attack_Category_x'].values
 
num_classes=7
 
# Encode target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 
 
# Split the data into training and testing sets for global evaluation later
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
 
# Step 2: Simulate 10 clients by splitting the training data
clients = 10
client_data = []
 
for i in range(clients):
    X_client, _, y_client, _ = train_test_split(X_train_full, y_train_full, test_size=0.8, random_state=i)
    client_data.append((X_client, y_client))
 
# Step 3: Define different model architectures
def create_mlp_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
 
 
def create_dnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))  # Explicitly define the input shape using the Input layer
 
    # Adding more layers to deepen the model
    model.add(Dense(512, activation='relu'))  # First hidden layer with 512 neurons
    model.add(BatchNormalization())
    model.add(Dropout(0.3))  # Increased dropout rate for regularization
 
    model.add(Dense(256, activation='relu'))  # Second hidden layer with 256 neurons
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
 
    model.add(Dense(128, activation='relu'))  # Third hidden layer with 128 neurons
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
 
    model.add(Dense(64, activation='relu'))  # Fourth hidden layer with 64 neurons
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
 
    model.add(Dense(num_classes, activation='softmax'))  # Output layer for multi-class classification
 
    model.compile(optimizer='AdamW', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
    return model
 
def create_autoencoder(input_shape):
    input_layer = Input(shape=(input_shape,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    output_layer = Dense(input_shape, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
 
# Step 4: Federated Learning Simulation
rounds = 2
epochs = 10
 
# Assign different global models to clusters
global_models = {
    "mlp": create_mlp_model(X_train_full.shape[1], num_classes),
    "dnn": create_dnn_model(X_train_full.shape[1], num_classes),
}
 
# Assign clients to model types
model_types = ["mlp", "mlp", "mlp", "mlp", "mlp", "dnn", "dnn", "dnn", "dnn", "dnn"]
# model_types = ["mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "dnn", "dnn", "dnn", "dnn", "dnn","dnn", "dnn", "dnn", "dnn", "dnn"]
# model_types = ["mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp", "mlp"]
 
for round_num in range(rounds):
    print(f"Round {round_num + 1}/{rounds}")
    local_weights = {"mlp": [], "dnn": []}
 
    for client_num, (X_client, y_client) in enumerate(client_data):
        model_type = model_types[client_num]
        print(f"Training client {client_num + 1} using {model_type.upper()}")
        local_model = globals()[f"create_{model_type}_model"](X_train_full.shape[1], num_classes)
        local_model.set_weights(global_models[model_type].get_weights())
        local_model.fit(X_client, y_client, epochs=epochs, batch_size=32, verbose=0)
        client_weights = local_model.get_weights()
 
        # # **Introduce a malicious client (e.g., client 0)**
        if client_num == 0:
            print("⚠️ Malicious client detected: Flipping signs of weight updates!")
            client_weights = [-np.array(w) for w in client_weights]  # Flip weight signs
 
        local_weights[model_type].append(client_weights)
 
    # Aggregate weights per model type
    for model_type in global_models.keys():
        if local_weights[model_type]:  # Ensure there are clients using this model
            new_weights = [
                np.mean([client_weights[layer] for client_weights in local_weights[model_type]], axis=0)
                for layer in range(len(global_models[model_type].get_weights()))
            ]
            global_models[model_type].set_weights(new_weights)
 
    # Evaluate each global model
    for model_type, global_model in global_models.items():
        y_pred_prob = global_model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"After Round {round_num + 1}/{rounds} - {model_type.upper()} Model Accuracy: {accuracy:.4f}")
 
 
