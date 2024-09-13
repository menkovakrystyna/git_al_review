import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from quality_check import evaluate_preprocessed_data


class dp:  # Bad class name, too short and unclear
    def __init__(self, d):  # Unclear parameter name
        self.d = d  # No explanation for self.d
        self.pd = None  # Poor variable name for processed data
        self.a = None  # Meaningless variable names
        self.b = None
        self.c = None
        self.e = None

    def p(self):  # Bad method name, unclear what it does
        X = self.d.drop('target', axis=1)  # No error handling, unclear variable name
        y = self.d['target']  # Unclear variable name

        # Variable names are not descriptive
        nf = X.select_dtypes(include=['int64', 'float64']).columns
        cf = X.select_dtypes(include=['object']).columns

        # Hard to understand what each step is doing
        nt = Pipeline(steps=[
            ('i', SimpleImputer(strategy='mean')),  # 'i' instead of 'imputer'
            ('s', StandardScaler())  # 's' instead of 'scaler'
        ])

        ct = Pipeline(steps=[
            ('i', SimpleImputer(strategy='constant', fill_value='missing')),  # 'i' instead of 'imputer'
            ('o', OneHotEncoder(handle_unknown='ignore'))  # 'o' instead of 'onehotencoder'
        ])

        p = ColumnTransformer(  # 'p' is too short for variable name
            transformers=[
                ('n', nt, nf),  # Unclear 'n' and 'c'
                ('c', ct, cf)
            ])

        Xp = p.fit_transform(X)  # No explanation for Xp, should be descriptive like 'X_processed'

        self.pd = pd.DataFrame(Xp)  # Using 'pd' here is confusing, since 'pd' is pandas

        # Bad variable names with no explanation on what they represent
        self.a, self.b, self.c, self.e = train_test_split(
            self.pd, y, test_size=0.2, random_state=42)

    def g(self):  # Unclear function name, no docstring
        return self.a, self.b, self.c, self.e  # Bad variable names


# No proper documentation, poor formatting and unclear comments
if __name__ == "__main__":
    d = pd.read_csv('../dummy_data.csv')  # Bad variable name for dataset

    # No clear explanation for why class is instantiated this way
    p = dp(d)

    # No explanation of what the method is doing
    p.p()

    # The get method doesn't explain what it retrieves
    X_train, X_test, y_train, y_test = p.g()

    # No explanation of what this function does
    evaluate_preprocessed_data(X_train, X_test)
