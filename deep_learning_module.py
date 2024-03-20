import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import logging
import joblib  # For saving and loading scikit-learn models

# Optionally suppress oneDNN optimization messages from TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepLearningModel:
    def __init__(self):
        self.model = None
        self.scaler_path = 'model_saved/scaler.pkl'  # Path to the saved scaler

    def load_data(self, filepath):
        try:
            data = pd.read_csv(filepath)
            if 'Stress_Level' not in data.columns:
                logging.error("'Stress_Level' column is missing in the dataset. Please ensure the dataset includes this column.")
                return None, None, None, None

            X = data.drop('Stress_Level', axis=1)  # Features
            y = data['Stress_Level']  # Target variable
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save the scaler for later use
            joblib.dump(scaler, self.scaler_path)
            
            logging.info(f"Data loaded and scaled successfully from {filepath}")
            return X_train_scaled, X_test_scaled, y_train, y_test
        except Exception as e:
            logging.error(f"Failed to load data from {filepath}: {e}", exc_info=True)
            return None, None, None, None

    def build_model(self, input_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_shape,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # Assuming 3 classes for stress level
        ])
        
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        logging.info("Model built and compiled successfully.")

    def train_model(self, X_train, X_test, y_train, y_test):
        if self.model is None:
            logging.error("Model has not been built.")
            return

        try:
            self.model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
            self.evaluate_model(X_test, y_test)
        except Exception as e:
            logging.error(f"Failed to train the model: {e}", exc_info=True)

    def evaluate_model(self, X_test, y_test):
        try:
            loss, accuracy = self.model.evaluate(X_test, y_test)
            logging.info(f"Model evaluation complete with accuracy: {accuracy}")
        except Exception as e:
            logging.error(f"Failed to evaluate the model: {e}", exc_info=True)

    def save_model(self, save_path='model_saved'):
        if self.model:
            try:
                # Ensure the save directory exists
                os.makedirs(save_path, exist_ok=True)
                # Specify the full path with the .h5 extension for the HDF5 format
                model_save_path = os.path.join(save_path, 'my_model.h5')
                self.model.save(model_save_path)
                logging.info(f"Model saved successfully at {model_save_path}.")
            except Exception as e:
                logging.error(f"Failed to save the model: {e}", exc_info=True)
        else:
            logging.error("No model to save.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train the deep learning model for stress level prediction.")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset CSV file.")
    args = parser.parse_args()

    model_instance = DeepLearningModel()
    X_train_scaled, X_test_scaled, y_train, y_test = model_instance.load_data(args.dataset_path)
    if X_train_scaled is not None:
        input_shape = X_train_scaled.shape[1]  # Number of features
        model_instance.build_model(input_shape)
        model_instance.train_model(X_train_scaled, X_test_scaled, y_train, y_test)
        model_instance.save_model()