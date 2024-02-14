import pandas as pd
from ucimlrepo import fetch_ucirepo

"""
Data source:
Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert.
(1988). Heart Disease.
UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X.
"""


def fetch_data():
    try:
        data = fetch_ucirepo(name='Heart Disease')
        X = data.data.features
        y = data.data.targets
        data = pd.concat([X, y], axis=1)
        data.to_csv('heart_disease.csv', index=False)
        print('Data successfully saved as heart_disease.csv.')

    except Exception as e:
        print(f'Error fetching data: {e}')


if __name__ == '__main__':
    fetch_data()
