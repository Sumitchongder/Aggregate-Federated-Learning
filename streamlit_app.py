import streamlit as st
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import joblib
import os
import tensorflow as tf

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
mnist_data = mnist.data.astype(np.float32)
mnist_target = mnist.target.astype(np.int32)

# Preprocess data
def preprocess(data, target):
    return data, target

# Define models for each client
def get_models(client_number):
    models = {
        1: [DecisionTreeClassifier()],
        2: [DecisionTreeClassifier(), GaussianNB()],
        3: [DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier()],
        4: [DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]
    }
    return models[client_number]

# Train client models
def train_client_models(client_number, num_epochs):
    train_data, train_labels = preprocess(mnist_data, mnist_target)
    models = get_models(client_number)

    accuracies = []
    build_times = []
    classification_times = []
    trained_models = []

    for model in models:
        start_time = time.time()
        for epoch in range(num_epochs):
            model.fit(train_data, train_labels)
        build_time = time.time() - start_time

        start_time = time.time()
        predictions = model.predict(train_data)
        classification_time = time.time() - start_time

        accuracy = accuracy_score(train_labels, predictions)
        accuracies.append(accuracy)
        build_times.append(build_time)
        classification_times.append(classification_time)

        trained_models.append(model)

    return accuracies, build_times, classification_times, trained_models

# Download trained models
def download_models(client_number, trained_models):
    save_dir = "./downloaded_models/"
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, model in enumerate(trained_models):
        model_filename = f"client{client_number}_model_{idx + 1}.joblib"
        model_path = os.path.join(save_dir, model_filename)
        joblib.dump(model, model_path)
    
    return save_dir

# Display dataset samples
def display_samples():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Display sample images
    num_samples = 7
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(train_images[i], cmap='gray')
        plt.title(f"Label: {train_labels[i]}")
        plt.axis('off')
    st.pyplot(plt)  # Display plots

def main():
    st.title("Aggregate Federated Learning")

    st.sidebar.title("Settings")
    num_epochs = st.sidebar.slider("Number of Epochs", 1, 50, 1)

    client_number = st.sidebar.selectbox("Select Client", [1, 2, 3, 4], index=0)

    if st.sidebar.button("Train Client"):
        st.sidebar.write(f"Training started for Client {client_number}...")
        accuracies, build_times, classification_times, trained_models = train_client_models(client_number, num_epochs)
        st.sidebar.write(f"Training completed for Client {client_number}!")

        st.subheader("Results")
        st.write(f"Accuracy: {max(accuracies)}")
        st.write(f"Build Time: {sum(build_times)} seconds")
        st.write(f"Classification Time: {sum(classification_times)} seconds")

        # Enable download button
        if st.button("Download Models"):
            save_dir = download_models(client_number, trained_models)
            st.write(f"Models for Client {client_number} downloaded successfully to: {save_dir}")

    st.sidebar.subheader("Sample Images")
    if st.sidebar.button("Display Dataset Samples"):
        display_samples()

if __name__ == "__main__":
    main()





