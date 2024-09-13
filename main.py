import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from quality_check import evaluate_preprocessed_data

self.X_test, self.y_train, self.y_test



# Пример использования
if __name__ == "__main__":
    # Fetch dummy data
    dummy_data = pd.read_csv('../dummy_data.csv')

    # Инициализация класса DataPreprocessor
    preprocessor = DataPreprocessor(dummy_data)

    # Выполнение предварительной обработки данных
    preprocessor.preprocess()

    # Получение обработанных данных
    X_train, X_test, y_train, y_test = preprocessor.get_processed_data()

    # This should fail due to missing values, non-numeric data, and inconsistent feature sets
    evaluate_preprocessed_data(X_train, X_test)
