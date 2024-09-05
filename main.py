import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from quality_check import evaluate_preprocessed_data


class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        """
        Конструктор для инициализации DataPreprocessor с данными.

        :param data: DataFrame с исходными данными
        """
        self.data = data
        self.preprocessed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        """
        Метод для выполнения предварительной обработки данных:
        - Заполнение пропущенных значений
        - Кодирование категориальных переменных
        - Масштабирование данных
        - Разделение на обучающий и тестовый наборы
        """
        # Разделение признаков и целевой переменной
        X = self.data.drop('target', axis=1)
        y = self.data['target']

        # Создание трансформеров для числовых и категориальных данных
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Композиция трансформеров
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Создание и запуск пайплайна
        X_processed = preprocessor.fit_transform(X)

        # Сохранение обработанных данных
        self.preprocessed_data = pd.DataFrame(X_processed)

        # Разделение на обучающий и тестовый наборы
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.preprocessed_data, y, test_size=0.2, random_state=42)

    def get_processed_data(self):
        """
        Возвращает обучающие и тестовые данные.

        :return: X_train, X_test, y_train, y_test
        """
        return self.X_train, self.X_test, self.y_train, self.y_test



# Пример использования
if __name__ == "__main__":
    # Fetch dummy data
    dummy_data = pd.read_csv('dummy_data.csv')

    # Инициализация класса DataPreprocessor
    preprocessor = DataPreprocessor(dummy_data)

    # Выполнение предварительной обработки данных
    preprocessor.preprocess()

    # Получение обработанных данных
    X_train, X_test, y_train, y_test = preprocessor.get_processed_data()

    evaluate_preprocessed_data(X_train, X_test)




