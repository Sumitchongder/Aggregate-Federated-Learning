import streamlit as st
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import os

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

def train_model(max_depth, min_samples_split):
    start_time = time.time()
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    clf.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    start_time = time.time()
    y_pred = clf.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, accuracy, train_time, inference_time

st.title('Federated Learning App')

# Hyperparameter tuning sidebar
st.sidebar.title('Hyperparameter Tuning')
max_depth = st.sidebar.slider('Max Depth', 1, 20, 1)
min_samples_split = st.sidebar.slider('Min Samples Split', 2, 100, 2)

model = None

# Train Client 1 button
if st.button('Train Client 1'):
    model, accuracy, train_time, inference_time = train_model(max_depth, min_samples_split)
    output_text = f"""
        Accuracy: {accuracy}
        Training Time: {train_time}
        Inference Time: {inference_time}
        Model trained and evaluated successfully!
    """
    output_elem = st.empty()
    output_elem.text(output_text)

# Download Model for Client 1 button
if model is not None:
    if st.button('Download Model for Client 1'):
        # Specify the path to the Downloads folder
        download_path = os.path.join(os.path.expanduser("~"), "Downloads", "client1_model.joblib")
        joblib.dump(model, download_path)
        st.write('Model downloaded successfully! It should be in your Downloads folder.')

# Display graphs
if st.button('Show Graphs'):
    # Since DecisionTreeClassifier doesn't involve epochs, we'll just plot random data for demonstration
    epochs = 10
    accuracies = np.random.rand(epochs) * 100
    build_times = np.random.rand(epochs) * 10
    inference_times = np.random.rand(epochs) * 5
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    
    axs[0].plot(range(1, epochs+1), accuracies, label='Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Accuracy over Epochs')
    axs[0].set_xlabel('Epochs')
    
    axs[1].plot(range(1, epochs+1), build_times, label='Build Time')
    axs[1].set_ylabel('Build Time (s)')
    axs[1].set_title('Build Time over Epochs')
    axs[1].set_xlabel('Epochs')
    
    axs[2].plot(range(1, epochs+1), inference_times, label='Inference Time')
    axs[2].set_ylabel('Inference Time (s)')
    axs[2].set_title('Inference Time over Epochs')
    axs[2].set_xlabel('Epochs')
    
    plt.tight_layout()
    st.pyplot(fig)
