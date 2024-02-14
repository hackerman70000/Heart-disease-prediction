import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_fetch import fetch_data


def map_categorical_values(df):
    mapping = {
        'sex': {1: 'male', 0: 'female'},
        'cp': {1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'},
        'fbs': {1: 'true', 0: 'false'},
        'restecg': {0: 'normal', 1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy'},
        'exang': {1: 'yes', 0: 'no'},
        'slope': {1: 'upsloping', 2: 'flat', 3: 'downsloping'},
        'thal': {3: 'normal', 6: 'fixed defect', 7: 'reversible defect'},
        'target': {1: 'heart disease', 0: 'no heart disease'}
    }

    df.replace(mapping, inplace=True)
    return df


def save_image(fig, filename):
    if not os.path.exists('images'):
        os.makedirs('images')
    filepath = os.path.join('images', filename)
    fig.savefig(filepath)


def process_target(df):
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.drop('num', axis=1, inplace=True)
    return df


def print_info(df):
    pd.set_option('display.max_columns', None)
    print('Dataframe shape:', df.shape, '\n')
    print('Dataframe sample: \n', df.sample(5), '\n')
    print('Dataframe duplicates: \n', df.duplicated().sum(), '\n')
    print('Dataframe types: \n', df.dtypes, '\n')
    print('Dataframe describe: \n', df.describe().T, '\n')

    # Missing values in 'ca' [4] and 'thal' [2] columns
    print('Missing values: \n', df.isnull().sum(), '\n')


def plot_histograms(df):
    df = map_categorical_values(df)
    columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    plt.figure(figsize=(20, 10))
    plt.suptitle('Heart disease incidence by different features', fontsize=20)

    for i, column in enumerate(columns, start=1):
        plt.subplot(3, 2, i)
        sns.histplot(x=column, hue='target', data=df, palette='viridis', kde=True)
        plt.title(column)

    save_image(plt, 'histplot.png')


def plot_count(df):
    df = map_categorical_values(df)
    columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.suptitle('Heart disease incidence by different features', fontsize=20)

    for i, column in enumerate(columns, start=1):
        plt.subplot(2, 4, i)
        sns.countplot(x='target', hue=column, data=df, palette='viridis')
        plt.title(column)

    save_image(plt, 'countplot.png')


def plot_correlation(df):
    plt.figure(figsize=(15, 10))
    plt.suptitle('Correlation between features', fontsize=20)
    sns.heatmap(df.corr(), annot=True, cmap='viridis')
    save_image(plt, 'correlation.png')


def pairplot(df):
    pairplot = sns.pairplot(df, hue='target', palette='viridis')
    save_image(pairplot, 'pairplot.png')


def plot_distribution(df):
    df = map_categorical_values(df)
    plt.figure(figsize=(20, 10))
    plt.suptitle('Density distribution of features', fontsize=20)
    for i, col in enumerate(df.columns, 1):
        plt.subplot(5, 3, i)
        plt.title(col)
        sns.histplot(df[col], kde=True)
        plt.tight_layout()
        plt.plot()

    save_image(plt, 'distribution.png')


def plots(df):
    plot_correlation(df)
    pairplot(df)
    plot_histograms(df)
    plot_count(df)
    plot_distribution(df)
    plt.show()


def detect_outliers(df):
    for col_name, col_values in df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].items():
        q1 = col_values.quantile(0.25)
        q3 = col_values.quantile(0.75)
        IRQ = q3 - q1
        outliers = col_values[(col_values <= q1 - 1.5 * IRQ) | (col_values >= q3 + 1.5 * IRQ)]
        percentage_outliers = len(outliers) * 100.0 / len(df)
        print("Column {} outliers = {}%".format(col_name, round(percentage_outliers, 2)))


if __name__ == '__main__':
    try:
        df = pd.read_csv('heart_disease.csv')
    except FileNotFoundError:
        fetch_data()
        try:
            df = pd.read_csv('heart_disease.csv')
        except FileNotFoundError:
            print('Error fetching data. Please try again.')
            exit(1)

    df = process_target(df)
    print_info(df)
    df = df.dropna()
    detect_outliers(df)
    plots(df)
