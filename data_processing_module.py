import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import logging
import os
from deep_learning_module import DeepLearningModel
import joblib  # For saving and loading scikit-learn models
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessingModule:
    def __init__(self):
        self.model_path = 'model_saved/my_model.h5'  # Path to the saved deep learning model
        self.scaler_path = 'model_saved/scaler.pkl'  # Path to the saved scaler
        self.random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model = self.load_model()
        self.scaler = self.load_scaler()
        self.deep_learning_model = DeepLearningModel()

    def validate_input(self, hrv, qrs_complex, rr_intervals, frequency_domain_features):
        try:
            inputs = np.array([float(hrv), float(qrs_complex), float(rr_intervals), float(frequency_domain_features)])
            logging.info("Input validation successful.")
            return inputs.reshape(1, -1), True  # Reshape for a single prediction
        except ValueError as e:
            logging.error("Input validation failed: ", exc_info=True)
            return None, False

    def preprocess_data(self, inputs):
        if self.scaler is not None:
            try:
                # Scale the data using the pre-fitted scaler
                scaled_data = self.scaler.transform(inputs)
                logging.info("Data preprocessing successful.")
                return scaled_data
            except Exception as e:
                logging.error("Error during data preprocessing: ", exc_info=True)
                return None
        else:
            logging.error("Scaler not loaded.")
            return None

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                model = tf.keras.models.load_model(self.model_path)
                logging.info("Model loaded successfully.")
                return model
            except Exception as e:
                logging.error("Failed to load model: ", exc_info=True)
                return None
        else:
            logging.info("Model file does not exist. Model needs to be trained.")
            return None

    def load_scaler(self):
        if os.path.exists(self.scaler_path):
            try:
                scaler = joblib.load(self.scaler_path)
                logging.info("Scaler loaded successfully.")
                return scaler
            except Exception as e:
                logging.error("Failed to load scaler: ", exc_info=True)
                return None
        else:
            logging.error("Scaler file does not exist. Scaler needs to be fitted and saved.")
            return None

    def fit_and_save_scaler(self, X):
        try:
            scaler = StandardScaler()
            scaler.fit(X)
            joblib.dump(scaler, self.scaler_path)
            logging.info("Scaler fitted and saved successfully.")
        except Exception as e:
            logging.error("Failed to fit and save scaler: ", exc_info=True)

    def train_model(self, X, y, best_params=None):
        try:
            # Splitting the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Update the classifier with best parameters if provided
            if best_params:
                self.random_forest_classifier.set_params(**best_params)
            
            # Initialize a Pipeline with feature selection and model training
            pipeline = Pipeline([
                ('scaling', StandardScaler()),
                ('feature_selection', SelectKBest(f_classif, k=4)),  # Select the top 4 features
                ('classification', self.random_forest_classifier)
            ])
             
            # Fit the pipeline
            pipeline.fit(X_train, y_train)
             
            # Evaluate the model
            predictions = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logging.info(f"Random Forest model trained successfully with accuracy: {accuracy}")
             
            # Save the trained model
            joblib.dump(pipeline, self.model_path.replace('.h5', '_rf_pipeline.pkl'))
            logging.info(f"Model saved to {self.model_path.replace('.h5', '_rf_pipeline.pkl')}.")
        except Exception as e:
            logging.error("Error during Random Forest model training: ", exc_info=True)

    def train_deep_learning_model(self, filepath):
        try:
            X_train, X_test, y_train, y_test = self.deep_learning_model.load_data(filepath)
            input_shape = X_train.shape[1]
            self.deep_learning_model.build_model(input_shape)
            self.deep_learning_model.train_model(X_train, X_test, y_train, y_test)
            self.deep_learning_model.save_model()
            logging.info("Deep learning model trained and saved successfully.")
        except Exception as e:
            logging.error(f"Error during deep learning model training: {e}", exc_info=True)

    def classify_stress_level(self, processed_data):
        stress_level_labels = {0: 'low', 1: 'moderate', 2: 'high'}
        if self.model is None:
            logging.error("Model is not trained. Please train the model before classifying stress levels.")
            return None

        try:
            if isinstance(self.model, tf.keras.Model):
                # Preprocess data for deep learning model
                predicted_stress_level = np.argmax(self.model.predict(processed_data), axis=1)
            else:
                # Use RandomForest or other sklearn model
                predicted_stress_level = self.model.predict(processed_data)

            stress_level = stress_level_labels.get(predicted_stress_level[0], 'Unknown')
            logging.info(f"Stress level classification successful. Predicted stress level: {stress_level}")
            return stress_level
        except Exception as e:
            logging.error("Error during stress level classification: ", exc_info=True)
            return None