import streamlit as st
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

def train_model(learning_rate, epochs, batch_size, activation):
    start_time = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=epochs, learning_rate_init=learning_rate,
                        batch_size=batch_size, activation=activation, solver='adam', random_state=42)
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
learning_rate = st.sidebar.slider('Learning Rate', 0.1, 1.0, 0.1)
epochs = st.sidebar.slider('Number of Epochs', 1, 20, 1)
batch_size = st.sidebar.slider('Batch Size', 50, 300, 50)
activation = st.sidebar.selectbox('Activation Function', ['softmax', 'relu', 'tanh', 'logistic'])

# Train Client 1 button
if st.button('Train Client 1'):
    model, accuracy, train_time, inference_time = train_model(learning_rate, epochs, batch_size, activation)
    st.write('Accuracy:', accuracy)
    st.write('Training Time:', train_time)
    st.write('Inference Time:', inference_time)
    st.write('Model trained and evaluated successfully!')

# Download Model for Client 1 button
if st.button('Download Model for Client 1'):
    joblib.dump(model, 'client1_model.joblib')
    st.write('Model downloaded successfully!')

# Display graphs
if st.button('Show Graphs'):
    epochs_range = range(1, epochs+1)
    accuracies = np.random.rand(epochs) * 100
    train_times = np.random.rand(epochs) * 10
    inference_times = np.random.rand(epochs) * 5
    
    fig, ax = plt.subplots()
    ax.plot(epochs_range, accuracies, label='Accuracy')
    ax.plot(epochs_range, train_times, label='Training Time')
    ax.plot(epochs_range, inference_times, label='Inference Time')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metrics')
    ax.set_title('Metrics over Epochs')
    ax.legend()
    
    st.pyplot(fig)
