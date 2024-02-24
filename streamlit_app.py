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

# Load MNIST data
mnist = fetch_openml('mnist_784', version=1)
mnist_data = mnist.data.astype(np.float32)
mnist_target = mnist.target.astype(np.int32)

# Preprocess data
def preprocess(data, target):
    return data, target

class ClientTrainingApp:
    def __init__(self):
        st.title("Aggregate Federated Learning")

        self.num_epochs = 1  # Initialize number of epochs

        # Create hyperparameters section
        st.sidebar.title("Hyperparameters")
        self.num_epochs = st.sidebar.slider("Number of Epochs", 1, 50, 1)
        self.learning_rate = st.sidebar.slider("Learning Rate", 0.00001, 1.0, 0.01)
        self.batch_size = st.sidebar.slider("Batch Size", 50, 300, 50)
        self.activation_function = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh", "Softmax"])

        # Button to start training
        if st.sidebar.button("Start Training"):
            self.train_clients()

    def train_clients(self):
        train_data, train_labels = preprocess(mnist_data, mnist_target)

        # Define models for each client
        models = {
            1: [DecisionTreeClassifier()],
            2: [DecisionTreeClassifier(), GaussianNB()],
            3: [DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier()],
            4: [DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier(), RandomForestClassifier()]
        }

        for client_number, client_models in models.items():
            st.write(f"Training started for Client {client_number}...")
            accuracies = []
            build_times = []
            classification_times = []
            trained_models = []

            for model in client_models:
                start_time = time.time()
                for epoch in range(self.num_epochs):
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

            self.plot_graphs(build_times, classification_times, accuracies, client_models, client_number)

            # Download trained models
            self.download_models(trained_models, client_number)

    def download_models(self, trained_models, client_number):
        # Specify the directory where models will be saved
        save_dir = "downloads"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Serialize and save the trained models
        for idx, model in enumerate(trained_models):
            model_filename = f"client{client_number}_model_{idx + 1}.joblib"
            model_path = os.path.join(save_dir, model_filename)
            joblib.dump(model, model_path)

        st.write(f"Model for Client {client_number} downloaded successfully!")

    def plot_graphs(self, build_times, classification_times, accuracies, models, client_number):
        st.write(f"### Results for Client {client_number}")

        st.write("#### Accuracy")
        st.bar_chart({type(model).__name__: accuracy for model, accuracy in zip(models, accuracies)})

        st.write("#### Build Time")
        st.bar_chart({type(model).__name__: build_time for model, build_time in zip(models, build_times)})

        st.write("#### Classification Time")
        st.bar_chart({type(model).__name__: classification_time for model, classification_time in zip(models, classification_times)})

if __name__ == "__main__":
    app = ClientTrainingApp()





