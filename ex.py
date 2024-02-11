from logging import warning
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load the CSV file
data = pd.read_csv("hour.csv")

# Display file names in 'input' directory
for dirname, _, hour in os.walk('input'):
    for filename in hour:
        print(os.path.join(dirname, filename))

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) // nGraphPerRow  # Corrected division operator
    plt.figure(num=None, figsize=(6 * nGraphPerRow, 8 * nGraphRow), dpi=80, facecolor='w', edgecolor='k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if not np.issubdtype(type(columnDf.iloc[0]), np.number):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation=90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    plt.show()

def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna(axis='columns')  # 'columns' instead of 'columns'
    df = df[[col for col in df if df[col].nunique() > 1]]
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis='columns')  # 'columns' instead of 'columns'
    df = df[[col for col in df if df[col].nunique() > 1]]
    columnNames = list(df)
    if len(columnNames) > 10:
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*np.triu_indices_from(ax, k=1)):  # Removed 'plt.'
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center',
                          va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = 1000
df1 = pd.read_csv('hour.csv', delimiter=',', nrows=nRowsRead)
df1.dataframeName = 'hour.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

print(df1.head(5))

# Drop columns from 'data' DataFrame
data = data.drop(['TIME'], axis=1)

print(data)

print(data.isnull().sum())

print(data.info())

# Convert categorical variables into dummy variables
data = pd.get_dummies(data, drop_first=True)
data = data.astype(float)
print(data)

print(data['dayhour'].value_counts())
print(data['cow'].value_counts())
print(data['avgtotalmotion'].value_counts())
print(data['lactation_number_in_data'].value_counts())

# Countplot for 'dayhour' variable using Seaborn
plt.figure(figsize=(8, 2))
sns.countplot(x='dayhour', data=data)
plt.show()

# Countplot for 'cow' variable using Seaborn
plt.figure(figsize=(10, 5))
sns.boxenplot(x='cow', data=data)
plt.show()

# Countplot for 'avgtotalmotion' variable using Seaborn
plt.figure(figsize=(8, 2))
sns.scatterplot(x='avgtotalmotion', data=data)
plt.show()

# Countplot for 'lactation_number_in_data' variable using Seaborn
plt.figure(figsize=(8, 2))
sns.countplot(x='lactation_number_in_data', data=data)
plt.show()

plotScatterMatrix(df1, 20, 10)
plt.show()

plotCorrelationMatrix(df1, 20)
plt.show()