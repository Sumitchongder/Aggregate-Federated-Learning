import streamlit as st
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import os

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
mnist_data = mnist.data.astype(np.float32)
mnist_target = mnist.target.astype(np.int32)

# Preprocess data
def preprocess(data, target):
    return data, target

# Function to train client models
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
    
    return build_times, classification_times, accuracies, trained_models

def download_model(client_number, trained_models):
    # Specify the directory where models will be saved
    save_dir = "C:/Users/Sumit/Downloads/"

    # Serialize and save the trained model
    for idx, model in enumerate(trained_models):
        model_filename = f"client{client_number}_model_{idx + 1}.joblib"
        model_path = os.path.join(save_dir, model_filename)
        joblib.dump(model, model_path)

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
    st.pyplot()  # Display plots using Streamlit

# Streamlit app
st.title("Aggregate Federated Learning")

# Number of epochs dropdown
num_epochs = st.selectbox("Number of Epochs", list(range(1, 51)), index=0)

# Train clients
if st.button("Train Client 1"):
    build_times, classification_times, accuracies, trained_models = train_client(1, num_epochs)
    st.write(f"Accuracy: {max(accuracies)}")
    st.write(f"Build Time: {sum(build_times)}")
    st.write(f"Classification Time: {sum(classification_times)}")
    download_model(1, trained_models)

if st.button("Train Client 2"):
    build_times, classification_times, accuracies, trained_models = train_client(2, num_epochs)
    st.write(f"Accuracy: {max(accuracies)}")
    st.write(f"Build Time: {sum(build_times)}")
    st.write(f"Classification Time: {sum(classification_times)}")
    download_model(2, trained_models)

if st.button("Train Client 3"):
    build_times, classification_times, accuracies, trained_models = train_client(3, num_epochs)
    st.write(f"Accuracy: {max(accuracies)}")
    st.write(f"Build Time: {sum(build_times)}")
    st.write(f"Classification Time: {sum(classification_times)}")
    download_model(3, trained_models)

if st.button("Train Client 4"):
    build_times, classification_times, accuracies, trained_models = train_client(4, num_epochs)
    st.write(f"Accuracy: {max(accuracies)}")
    st.write(f"Build Time: {sum(build_times)}")
    st.write(f"Classification Time: {sum(classification_times)}")
    download_model(4, trained_models)

# Display dataset samples button
if st.button("Display Dataset Samples"):
    display_samples()
