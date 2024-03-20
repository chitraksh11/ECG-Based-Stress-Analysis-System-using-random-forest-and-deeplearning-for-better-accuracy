# ECG Based Stress Analysis System

The ECG Based Stress Analysis System is a desktop application developed in Python, designed to help users analyze their stress levels through ECG-related parameters such as Heart Rate Variability (HRV), QRS Complex, R-R Intervals, and Frequency Domain Features. Utilizing machine learning models, the application categorizes stress levels into normal, moderate, and high, providing a graphical representation of stress variations over time.

# Overview

The application architecture comprises three main modules:
- # User Interface Module: Facilitates the manual input of ECG parameters by the user.
- # Data Processing and Analysis Module**: Employs machine learning models to process and classify stress levels.
- # Graphical Output Module: Generates visual representations of the analysis.

Technologies used include Scikit-learn, TensorFlow or PyTorch for machine learning, and Matplotlib and Seaborn for graphical outputs.

# Features

- # Manual Data Input: Users manually input ECG-related parameters.
- # ECG Data Analysis: The application classifies stress levels using machine learning models.
- **Graphical Representation**: Graphical outputs help users track their stress levels over time.

# Getting started

# Requirements

- Python 3.8 or newer
- PyQt5
- Scikit-learn, TensorFlow/PyTorch
- Pandas, NumPy
- Matplotlib, Seaborn

# Quickstart

1. Install the required packages with `pip install -r requirements.txt`.
2. To train the machine learning models, execute `python train_model_script.py`. Before running the script, ensure you have a dataset prepared for training. Update the `dataset_path` variable in `train_model_script.py` with the path to your dataset. This script trains both RandomForest and deep learning models and logs their accuracy.
3. Launch the application by running `ui_module.py`. In this interface, you can manually input ECG-related parameters for stress analysis.
4. The application provides stress level analysis and graphical representations, allowing you to view your stress levels over time.
