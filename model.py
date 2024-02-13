import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def process_target(df):
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.drop('num', axis=1, inplace=True)
    return df


def results(y_test, y_pred):
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    mse = mean_squared_error(y_test, y_pred)
    print_results(precision, recall, accuracy, confusion_matrix, mse)


def print_results(precision, recall, accuracy, confusion_matrix, mse):
    print('Precision:', round(precision, 3))
    print('Recall:', round(recall, 3))
    print('Accuracy:', round(accuracy, 3))
    print('Confusion matrix: \n', confusion_matrix)
    print('Mean squared error:', round(mse, 3))


df = pd.read_csv('heart_disease.csv')
df = df.dropna()
df.reset_index(drop=True, inplace=True)
df = process_target(df)

X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

results(y_test, y_pred)
