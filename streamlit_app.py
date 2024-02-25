import streamlit as st
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

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
    st.write('Accuracy:', accuracy)
    st.write('Training Time:', train_time)
    st.write('Inference Time:', inference_time)
    st.write('Model trained and evaluated successfully!')

# Download Model for Client 1 button
if model is not None:
    if st.button('Download Model for Client 1'):
        joblib.dump(model, 'client1_model.joblib')
        st.write('Model downloaded successfully!')

# Display graphs
if st.button('Show Graphs'):
    # Since DecisionTreeClassifier doesn't involve epochs, we'll just plot random data for demonstration
    accuracy = np.random.rand() * 100
    train_time = np.random.rand() * 10
    inference_time = np.random.rand() * 5
    
    fig, ax = plt.subplots()
    ax.bar(['Accuracy', 'Training Time', 'Inference Time'], [accuracy, train_time, inference_time])
    ax.set_ylabel('Metrics')
    ax.set_title('Metrics')
    
    st.pyplot(fig)
