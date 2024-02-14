import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from tabulate import tabulate

from data_fetch import fetch_data


def process_target(df):
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.drop('num', axis=1, inplace=True)
    return df


def results(y_test, y_pred):
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print_results(precision, recall, accuracy, mse)


def print_results(precision, recall, accuracy, mse):
    table = [
        ["Precision", round(precision, 3)],
        ["Recall", round(recall, 3)],
        ["Accuracy", round(accuracy, 3)],
        ["Mean Squared Error", round(mse, 3)]
    ]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))


def handle_outliers(df):
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df


def XGBoost_pipeline(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler()),
        ('clf', xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=1,
            n_estimators=50,
            colsample_bytree=0.5
        ))
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print('Results for XGBoost pipeline:')
    print("Mean CV accuracy:", round(np.mean(cv_scores), 3))
    results(y_test, y_pred)


def SVM_pipeline(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler()),
        ('clf', SVC())
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print('\n', 'Results for SVM pipeline:')
    print("Mean CV accuracy:", round(np.mean(cv_scores), 3))
    results(y_test, y_pred)


def LR_pipeline(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print('\n', 'Results for Logistic Regression pipeline:')
    print("Mean CV accuracy:", round(np.mean(cv_scores), 3))
    results(y_test, y_pred)


def KN_pipeline(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier())
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print('\n', 'Results for KNeighbors pipeline:')
    print("Mean CV accuracy:", round(np.mean(cv_scores), 3))
    results(y_test, y_pred)


if __name__ == "__main__":
    try:
        df = pd.read_csv('heart_disease.csv')
    except FileNotFoundError:
        fetch_data()
        try:
            df = pd.read_csv('heart_disease.csv')
        except FileNotFoundError:
            print('Error fetching data. Please try again.')
            exit(1)

    df = df.dropna().reset_index(drop=True)
    df = process_target(df)
    df = handle_outliers(df)

    X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
            'thal']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    KN_pipeline(X_train, X_test, y_train, y_test)
    LR_pipeline(X_train, X_test, y_train, y_test)
    SVM_pipeline(X_train, X_test, y_train, y_test)
    XGBoost_pipeline(X_train, X_test, y_train, y_test)
