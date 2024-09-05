import numpy as np
import pandas as pd


def evaluate_preprocessed_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Function to evaluate whether the preprocessed data is suitable for model ingestion.
    It checks:
    - No missing values
    - All features are numeric (categorical ones are one-hot encoded)
    - Consistent number of features between training and test sets
    """

    # Check for missing values in the training and test sets
    if np.any(pd.isnull(X_train)) or np.any(pd.isnull(X_test)):
        print("Evaluation Failed: Preprocessed data contains missing values.")
        return False

    # Check if all columns are numeric (categorical features should have been one-hot encoded)
    if not X_train.dtypes.apply(np.issubdtype, args=(np.number,)).all() or \
            not X_test.dtypes.apply(np.issubdtype, args=(np.number,)).all():
        print("Evaluation Failed: Preprocessed data contains non-numeric values.")
        return False

    # Ensure the number of features in training and testing sets are the same
    if X_train.shape[1] != X_test.shape[1]:
        print("Evaluation Failed: Inconsistent number of features between training and testing sets.")
        return False

    print("Evaluation Passed: Preprocessed data is ready for model ingestion.")
    return True
