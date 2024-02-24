import tkinter as tk
from tkinter import ttk
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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

class ClientTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Aggregate Federated Learning")
        self.root.geometry("800x600")
        self.root.configure(bg='lightgrey')

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10)

        # Add tabs
        self.create_training_tab()
        self.create_hyperparameters_tab()

        # Initialize lists for storing results
        self.build_times = []
        self.classification_times = []
        self.accuracies = []
        self.client_labels = []
        self.num_epochs = 1  # Initialize number of epochs

        # Initialize download model buttons
        self.download_model_buttons = []
        for i in range(1, 5):
            button = tk.Button(root, text=f"Download Model from Client {i}", font=("Arial", 12), command=lambda i=i: self.download_model(i), state=tk.DISABLED)
            button.pack()
            self.download_model_buttons.append(button)
            
        # Button to display dataset samples
        self.display_button = tk.Button(root, text="Display Dataset Samples", font=("Arial", 12), command=self.display_samples)
        self.display_button.pack(pady=10)

    def create_training_tab(self):
        # Create a frame for the training tab
        training_tab = tk.Frame(self.notebook, bg='lightgrey')
        self.notebook.add(training_tab, text="Training")

        self.accuracy_label = tk.Label(training_tab, text="Accuracy:", font=("Arial", 12), bg='lightgrey')
        self.accuracy_label.pack()

        self.build_time_label = tk.Label(training_tab, text="Build Time:", font=("Arial", 12), bg='lightgrey')
        self.build_time_label.pack()

        self.classification_time_label = tk.Label(training_tab, text="Classification Time:", font=("Arial", 12), bg='lightgrey')
        self.classification_time_label.pack()

        self.train_client_button1 = tk.Button(training_tab, text="Train Client 1", font=("Arial", 12), command=lambda: self.train_client(1))
        self.train_client_button1.pack()

        self.train_client_button2 = tk.Button(training_tab, text="Train Client 2", font=("Arial", 12), command=lambda: self.train_client(2))
        self.train_client_button2.pack()

        self.train_client_button3 = tk.Button(training_tab, text="Train Client 3", font=("Arial", 12), command=lambda: self.train_client(3))
        self.train_client_button3.pack()

        self.train_client_button4 = tk.Button(training_tab, text="Train Client 4", font=("Arial", 12), command=lambda: self.train_client(4))
        self.train_client_button4.pack()

        self.training_status_label = tk.Label(training_tab, text="", font=("Arial", 12), bg='lightgrey')
        self.training_status_label.pack()
        
        

    def create_hyperparameters_tab(self):
        # Create a frame for the hyperparameters tab
        hyperparameters_tab = tk.Frame(self.notebook, bg='lightgrey')
        self.notebook.add(hyperparameters_tab, text="Hyperparameters")

        # Create dropdown menus for hyperparameters
        self.learning_rate_label = tk.Label(hyperparameters_tab, text="Learning Rate:", font=("Arial", 12), bg='lightgrey')
        self.learning_rate_label.grid(row=0, column=0, padx=10, pady=5)
        self.learning_rate_var = tk.StringVar()
        self.learning_rate_dropdown = ttk.Combobox(hyperparameters_tab, textvariable=self.learning_rate_var, state="readonly")
        self.learning_rate_dropdown['values'] = [f"{i:.2f}" for i in np.arange(0.00001, 1.01, 0.01)]
        self.learning_rate_dropdown.current(0)  # Set default value
        self.learning_rate_dropdown.grid(row=0, column=1, padx=10, pady=5)

        self.num_epochs_label = tk.Label(hyperparameters_tab, text="Number of Epochs:", font=("Arial", 12), bg='lightgrey')
        self.num_epochs_label.grid(row=1, column=0, padx=10, pady=5)
        self.num_epochs_var = tk.IntVar()
        self.num_epochs_dropdown = ttk.Combobox(hyperparameters_tab, textvariable=self.num_epochs_var, state="readonly")
        self.num_epochs_dropdown['values'] = list(range(1, 51))
        self.num_epochs_dropdown.current(0)  # Set default value
        self.num_epochs_dropdown.grid(row=1, column=1, padx=10, pady=5)

        self.batch_size_label = tk.Label(hyperparameters_tab, text="Batch Size:", font=("Arial", 12), bg='lightgrey')
        self.batch_size_label.grid(row=2, column=0, padx=10, pady=5)
        self.batch_size_var = tk.IntVar()
        self.batch_size_dropdown = ttk.Combobox(hyperparameters_tab, textvariable=self.batch_size_var, state="readonly")
        self.batch_size_dropdown['values'] = list(range(50, 301))
        self.batch_size_dropdown.current(0)  # Set default value
        self.batch_size_dropdown.grid(row=2, column=1, padx=10, pady=5)

        self.activation_function_label = tk.Label(hyperparameters_tab, text="Activation Function:", font=("Arial", 12), bg='lightgrey')
        self.activation_function_label.grid(row=3, column=0, padx=10, pady=5)
        self.activation_function_var = tk.StringVar()
        self.activation_function_dropdown = ttk.Combobox(hyperparameters_tab, textvariable=self.activation_function_var, state="readonly")
        self.activation_function_dropdown['values'] = ["ReLU", "Sigmoid", "Tanh", "Softmax"]
        self.activation_function_dropdown.current(0)  # Set default value
        self.activation_function_dropdown.grid(row=3, column=1, padx=10, pady=5)

        # Bind the selection of number of epochs to update the variable
        self.num_epochs_dropdown.bind("<<ComboboxSelected>>", self.update_num_epochs)

    def update_num_epochs(self, event):
        self.num_epochs = self.num_epochs_var.get()

    def train_client(self, client_number):
        self.training_status_label.config(text=f"Training started for Client {client_number}...")
        self.root.update()  # Force update to immediately display the message
    
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
            for epoch in range(self.num_epochs):  # Use selected number of epochs
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
    
        self.trained_models = trained_models  # Assign trained models to the attribute
    
        self.plot_graphs(build_times, classification_times, accuracies, models[client_number], client_number)
        
        self.accuracy_label.config(text=f"Accuracy: {max(accuracies)}", fg='blue')
        self.build_time_label.config(text=f"Build Time: {sum(build_times)}", fg='blue')
        self.classification_time_label.config(text=f"Classification Time: {sum(classification_times)}", fg='blue')
    
        self.build_times.append(sum(build_times))
        self.classification_times.append(sum(classification_times))
        self.accuracies.append(max(accuracies))
        self.client_labels.append(f"Client {client_number}")
    
        # Enable corresponding download button
        self.download_model_buttons[client_number - 1].config(state=tk.NORMAL)

    def download_model(self, client_number):
        # Specify the directory where models will be saved
        save_dir = "C:/Users/Sumit/Downloads/"

        # Serialize and save the trained model
        for idx, model in enumerate(self.trained_models):
            model_filename = f"client{client_number}_model_{idx + 1}.joblib"
            model_path = os.path.join(save_dir, model_filename)
            joblib.dump(model, model_path)
        
        self.training_status_label.config(text=f"Model for Client {client_number} downloaded successfully!", fg='green')
        
    def display_samples(self):
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
        plt.show()  # Display plots non-blocking
        

    def plot_graphs(self, build_times, classification_times, accuracies, models, client_number):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        model_names = [type(model).__name__ for model in models]

        # Plot accuracy
        axes[0].bar(model_names, accuracies, color='blue')
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title(f'Accuracy for Client {client_number}')

        # Plot build time
        axes[1].bar(model_names, build_times, color='green')
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('Build Time')
        axes[1].set_title(f'Build Time for Client {client_number}')

        # Plot classification time
        axes[2].bar(model_names, classification_times, color='red')
        axes[2].set_xlabel('Models')
        axes[2].set_ylabel('Classification Time')
        axes[2].set_title(f'Classification Time for Client {client_number}')

        plt.tight_layout()

        # Print metrics below the graphs
        print(f"Accuracy for Client {client_number}: {accuracies}")
        print(f"Build Time for Client {client_number}: {build_times}")
        print(f"Classification Time for Client {client_number}: {classification_times}")

        plt.show()

root = tk.Tk()
gui = ClientTrainingGUI(root)
root.mainloop()
