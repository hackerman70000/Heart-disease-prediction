from ucimlrepo import fetch_ucirepo
import pandas as pd

def fetch_data():
    data = fetch_ucirepo(name='Heart Disease')
    X = data.data.features
    y = data.data.targets
    data = pd.concat([X, y], axis=1)
    data.to_csv('heart_disease.csv', index=False)

if __name__ == '__main__':
    fetch_data()
    data = pd.read_csv('heart_disease.csv')
    print(data.isnull().sum())


