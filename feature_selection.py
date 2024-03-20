import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_features_kbest(features, labels, k=10):
    """
    Selects the top k features that have the highest score with the chi-squared statistical test.

    :param features: DataFrame containing the feature columns.
    :param labels: Series containing the target variable.
    :param k: Number of top features to select.
    :return: DataFrame containing the selected feature columns.
    """
    try:
        selector = SelectKBest(score_func=chi2, k=k)
        selector.fit(features, labels)
        mask = selector.get_support(indices=True)
        selected_features = features.iloc[:, mask]
        logging.info(f"KBest feature selection completed. Selected features: {selected_features.columns.tolist()}")
        return selected_features
    except Exception as e:
        logging.error("Failed to select features with KBest.", exc_info=True)
        return None

def select_features_model_based(features, labels):
    """
    Selects features according to the importance weights of a model, in this case, RandomForestClassifier.

    :param features: DataFrame containing the feature columns.
    :param labels: Series containing the target variable.
    :return: DataFrame containing the selected feature columns.
    """
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, labels)
        selector = SelectFromModel(model, prefit=True)
        mask = selector.get_support(indices=True)
        selected_features = features.iloc[:, mask]
        logging.info(f"Model-based feature selection completed. Selected features: {selected_features.columns.tolist()}")
        return selected_features
    except Exception as e:
        logging.error("Failed to select features with model-based method.", exc_info=True)
        return None

if __name__ == "__main__":
    # Load your dataset here
    dataset_path = "path/to/your/dataset.csv"  # INPUT_REQUIRED {Update this path to your actual dataset path}
    try:
        data = pd.read_csv(dataset_path)
        X = data.drop('target_column', axis=1)  # Replace 'target_column' with the actual target column name in your dataset
        y = data['target_column']  # Replace 'target_column' with the actual target column name in your dataset

        # Select features using KBest
        selected_features_kbest = select_features_kbest(X, y, k=10)

        # Select features using Model-based method
        selected_features_model_based = select_features_model_based(X, y)
    except Exception as e:
        logging.error("Failed to load dataset or select features.", exc_info=True)