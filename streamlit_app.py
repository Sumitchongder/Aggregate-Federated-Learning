import streamlit as st
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import joblib
import os

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
mnist_data = mnist.data.astype(np.float32)
mnist_target = mnist.target.astype(np.int32)

# Preprocess data
def preprocess(data, target):
    return data, target

# Define model training function
def train_client(client_number, num_epochs):
    train_data, train_labels = preprocess(mnist_data, mnist_target)
    
    # Define models for each client
    models = {
        1: [DecisionTreeClassifier()],
        2: [DecisionTreeClassifier(), GaussianNB()],
        3: [DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier()],
        4: [DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]
    }
    
    accuracies = []
    build_times = []
    classification_times = []
    
    trained_models = []  # Store trained models for later use
    
    for model in models[client_number]:
        start_time = time.time()
        for epoch in range(num_epochs):  # Use selected number of epochs
            model.fit(train_data, train_labels)
        build_time = time.time() - start_time
        
        start_time = time.time()
        predictions = model.predict(train_data)
        classification_time = time.time() - start_time
        
        accuracy = accuracy_score(train_labels, predictions)
        accuracies.append(accuracy)
        build_times.append(build_time)
        classification_times.append(classification_time)

        trained_models.append(model)  # Store the trained model
    
    return accuracies, build_times, classification_times, trained_models

# Deployed Streamlit app code
st.title("Aggregate Federated Learning")

# Sidebar for hyperparameters
st.sidebar.title("Hyperparameters")
num_epochs = st.sidebar.slider("Number of Epochs", 1, 50, 1)

# Training
for i in range(1, 5):
    st.write(f"## Training Client {i}")
    accuracies, build_times, classification_times, trained_models = train_client(i, num_epochs)
    
    # Display metrics
    st.write(f"Accuracy: {max(accuracies)}")
    st.write(f"Build Time: {sum(build_times)}")
    st.write(f"Classification Time: {sum(classification_times)}")

    # Download models
    if st.button(f"Download Model for Client {i}"):
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)
        for idx, model in enumerate(trained_models):
            model_filename = f"client{i}_model_{idx + 1}.joblib"
            model_path = os.path.join(save_dir, model_filename)
            joblib.dump(model, model_path)
        st.success(f"Model for Client {i} downloaded successfully!")

# Display dataset samples
st.write("## Dataset Samples")
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
num_samples = 7
for i in range(num_samples):
    st.image(train_images[i], caption=f"Label: {train_labels[i]}", width=100)

