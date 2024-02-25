import streamlit as st
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
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

def train_model_nb():
    start_time = time.time()
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    start_time = time.time()
    y_pred = nb.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return nb, accuracy, train_time, inference_time

st.title('Federated Learning App')

# Hyperparameter tuning sidebar for Decision Tree
st.sidebar.title('Hyperparameter Tuning - Decision Tree')
max_depth_dt = st.sidebar.slider('Max Depth', 1, 20, 1)
min_samples_split_dt = st.sidebar.slider('Min Samples Split', 2, 100, 2)

# Hyperparameter tuning sidebar for Naive Bayes
st.sidebar.title('Hyperparameter Tuning - Naive Bayes')

model_dt = None
model_nb = None

# Train Client 1 button for Decision Tree
if st.button('Train Client 1 - Decision Tree'):
    model_dt, accuracy_dt, train_time_dt, inference_time_dt = train_model(max_depth_dt, min_samples_split_dt)
    output_text = f"""
        Decision Tree Model Metrics:
        Accuracy: {accuracy_dt}
        Training Time: {train_time_dt}
        Inference Time: {inference_time_dt}
        Decision Tree Model trained and evaluated successfully!
    """
    output_elem = st.empty()
    output_elem.text(output_text)

# Train Client 2 button for Decision Tree and Naive Bayes
if st.button('Train Client 2 - Decision Tree & Naive Bayes'):
    model_dt, accuracy_dt, train_time_dt, inference_time_dt = train_model(max_depth_dt, min_samples_split_dt)
    model_nb, accuracy_nb, train_time_nb, inference_time_nb = train_model_nb()
    
    output_text = f"""
        Decision Tree Model Metrics:
        Accuracy: {accuracy_dt}
        Training Time: {train_time_dt}
        Inference Time: {inference_time_dt}
        Decision Tree Model trained and evaluated successfully!
        
        Naive Bayes Model Metrics:
        Accuracy: {accuracy_nb}
        Training Time: {train_time_nb}
        Inference Time: {inference_time_nb}
        Naive Bayes Model trained and evaluated successfully!
    """
    output_elem = st.empty()
    output_elem.text(output_text)

# Download Model for Client 1 button for Decision Tree
if model_dt is not None:
    joblib.dump(model_dt, "client1_model_dt.joblib")
    with open("client1_model_dt.joblib", "rb") as f:
        model_binary = f.read()
    download_button = st.download_button(label="Download Decision Tree Model for Client 1", data=model_binary, file_name="client1_model_dt.joblib")

# Download Model for Client 2 button for Decision Tree and Naive Bayes
if model_dt is not None and model_nb is not None:
    joblib.dump(model_dt, "client2_model_dt.joblib")
    joblib.dump(model_nb, "client2_model_nb.joblib")
    with open("client2_model_dt.joblib", "rb") as f_dt, open("client2_model_nb.joblib", "rb") as f_nb:
        model_binary_dt = f_dt.read()
        model_binary_nb = f_nb.read()
    download_button_dt = st.download_button(label="Download Decision Tree Model for Client 2", data=model_binary_dt, file_name="client2_model_dt.joblib")
    download_button_nb = st.download_button(label="Download Naive Bayes Model for Client 2", data=model_binary_nb, file_name="client2_model_nb.joblib")

# Display graphs comparing Decision Tree and Naive Bayes
if st.button('Show Graphs'):
    # Since DecisionTreeClassifier doesn't involve epochs, we'll just plot random data for demonstration
    epochs = 10
    accuracies_dt = np.random.rand(epochs) * 100
    build_times_dt = np.random.rand(epochs) * 10
    inference_times_dt = np.random.rand(epochs) * 5
    
    accuracies_nb = np.random.rand(epochs) * 100
    build_times_nb = np.random.rand(epochs) * 10
    inference_times_nb = np.random.rand(epochs) * 5
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    axs[0, 0].bar(['Decision Tree', 'Naive Bayes'], [accuracy_dt, accuracy_nb])
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_title('Accuracy Comparison')
    
    axs[0, 1].bar(['Decision Tree', 'Naive Bayes'], [train_time_dt, train_time_nb])
    axs[0, 1].set_ylabel('Training Time (s)')
    axs[0, 1].set_title('Training Time Comparison')
    
    axs[1, 0].bar(['Decision Tree', 'Naive Bayes'], [inference_time_dt, inference_time_nb])
    axs[1, 0].set_ylabel('Inference Time (s)')
    axs[1, 0].set_title('Inference Time Comparison')
    
    axs[1, 1].bar(['Decision Tree', 'Naive Bayes'], [np.mean(build_times_dt), np.mean(build_times_nb)])
    axs[1, 1].set_ylabel('Mean Build Time (s)')
    axs[1, 1].set_title('Mean Build Time Comparison')
    
    axs[2, 0].plot(range(1, epochs+1), accuracies_dt, label='Decision Tree')
    axs[2, 0].plot(range(1, epochs+1), accuracies_nb, label='Naive Bayes')
    axs[2, 0].set_ylabel('Accuracy')
    axs[2, 0].set_title('Accuracy over Epochs')
    axs[2, 0].set_xlabel('Epochs')
    axs[2, 0].legend()
    
    axs[2, 1].plot(range(1, epochs+1), build_times_dt, label='Decision Tree')
    axs[2, 1].plot(range(1, epochs+1), build_times_nb, label='Naive Bayes')
    axs[2, 1].set_ylabel('Build Time (s)')
    axs[2, 1].set_title('Build Time over Epochs')
    axs[2, 1].set_xlabel('Epochs')
    axs[2, 1].legend()
    
    plt.tight_layout()
    st.pyplot(fig)

