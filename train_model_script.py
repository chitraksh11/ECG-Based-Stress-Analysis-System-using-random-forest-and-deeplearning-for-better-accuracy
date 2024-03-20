import logging
from data_processing_module import DataProcessingModule
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    logging.info("Starting model training process...")

    # Provide the path to your actual dataset.
    dataset_path = 'C:\\Users\\sharm\\Downloads\\project\\combined_stress_data.csv'  # INPUT_REQUIRED {Update this path to your actual dataset path if different}

    try:
        # Load dataset
        data = pd.read_csv(dataset_path)
        if data.empty:
            logging.error("The dataset is empty. Please check the dataset path and ensure it is correct.")
            return

        X = data.drop('Stress_Level', axis=1)
        y = data['Stress_Level']

        # Initialize DataProcessingModule
        dpm = DataProcessingModule()

        accuracy_info = ""  # Variable to hold accuracy information

        # Decide which model to train based on the availability of the train_model method
        if hasattr(dpm, 'train_model'):
            logging.info("Initiating RandomForestClassifier model training...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(dpm.random_forest_classifier, X, y, cv=5)
            accuracy_info = f"Cross-validation scores: {np.mean(cv_scores)}"
            logging.info(accuracy_info)
            
            # Hyperparameter tuning
            param_grid = {'n_estimators': [10, 100, 1000], 'max_features': ['sqrt', 'log2']}
            grid_search = GridSearchCV(dpm.random_forest_classifier, param_grid, cv=5)
            grid_search.fit(X, y)
            logging.info(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
            
            # Training the model with best parameters
            dpm.train_model(X, y, grid_search.best_params_)
            logging.info("RandomForestClassifier model training completed successfully.")

        elif hasattr(dpm, 'train_deep_learning_model'):
            logging.info("Initiating deep learning model training...")
            # For deep learning model training, the path to the dataset is needed
            dpm.train_deep_learning_model(dataset_path)
            logging.info("Deep learning model training completed successfully.")
        else:
            logging.error("The DataProcessingModule does not have the required training methods.")
    except Exception as e:
        logging.error("An error occurred during the model training process.", exc_info=True)

    # Log or save the accuracy information
    if accuracy_info:
        logging.info(f"Model training completed. {accuracy_info}")
    else:
        logging.info("Model training completed without accuracy evaluation.")

if __name__ == '__main__':
    train_model()