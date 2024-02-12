import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def mapping(df):
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


def plot_histograms(df):
    df = mapping(df)
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.suptitle('Heart disease incidence by different features', fontsize=20)

    plt.subplot(2, 2, 1)
    sns.histplot(x='age', hue='target', data=df, palette='viridis', kde=True)
    plt.title('age')

    plt.subplot(2, 2, 2)
    sns.histplot(x='trestbps', hue='target', data=df, palette='viridis', kde=True)
    plt.title('resting blood pressure')

    plt.subplot(2, 2, 3)
    sns.histplot(x='chol', hue='target', data=df, palette='viridis', kde=True)
    plt.title('cholesterol')

    plt.subplot(2, 2, 4)
    sns.histplot(x='thalach', hue='target', data=df, palette='viridis', kde=True)
    plt.title('maximum heart rate achieved')

    plt.savefig('images/histograms.png')
    plt.show()


def plot_count(df):
    df = mapping(df)
    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.suptitle('Heart disease incidence by different features', fontsize=20)

    plt.subplot(2, 4, 1)
    sns.countplot(x='target', hue='sex', data=df, palette='viridis')
    plt.title('sex')

    plt.subplot(2, 4, 2)
    sns.countplot(x='target', hue='cp', data=df, palette='viridis')
    plt.title('chest pain type')

    plt.subplot(2, 4, 3)
    sns.countplot(x='target', hue='fbs', data=df, palette='viridis')
    plt.title('fasting blood sugar')

    plt.subplot(2, 4, 4)
    sns.countplot(x='target', hue='restecg', data=df, palette='viridis')
    plt.title('resting electrocardiographic results')

    plt.subplot(2, 4, 5)
    sns.countplot(x='target', hue='exang', data=df, palette='viridis')
    plt.title('exercise induced angina')

    plt.subplot(2, 4, 6)
    sns.countplot(x='target', hue='slope', data=df, palette='viridis')
    plt.title('slope of the peak exercise ST segment')

    plt.subplot(2, 4, 7)
    sns.countplot(x='target', hue='thal', data=df, palette='viridis')
    plt.title('thal')

    plt.savefig('images/count_plots.png')
    plt.show()


def process_target(df):
    df['target'] = [1 if i > 0 else 0 for i in df['num']]
    df.drop('num', axis=1, inplace=True)
    print(df['target'].value_counts())
    return df


if __name__ == '__main__':
    df = pd.read_csv('heart_disease.csv')

    df = process_target(df)

    pd.set_option('display.max_columns', None)
    print('Dataframe shape:', df.shape, '\n')
    print('Dataframe sample: \n', df.sample(5), '\n')
    print('Dataframe duplicates: \n', df.duplicated().sum(), '\n')
    print('Dataframe types: \n', df.dtypes, '\n')
    print('Dataframe describe: \n', df.describe().T, '\n')

    # Missing values in 'ca' [4] and 'thal' [2] columns
    print('Missing values: \n', df.isnull().sum(), '\n')
    df = df.dropna()

    pairplot = sns.pairplot(df, hue='target', palette='viridis', height=3)
    if not os.path.exists('images'):
        os.makedirs('images')
        pairplot.savefig("images/pairplot.png")
        plt.close()

    plot_count(df)
    plot_histograms(df)
