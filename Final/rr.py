# Import module here
import numpy as np
import matplotlib as mt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as sts

# Loading and reading in Pandas
data = pd.read_csv('HR.csv')
data.head()

# Main setup
data_train, data_test = train_test_split(data, test_size = 0.1)

# Delete Salary and Department setup
#data_train = data.drop(['department','salary'], axis=1)
#data_test = data.drop(['department','salary'], axis=1)
del data_train['department']
del data_test['department']
del data_train['salary']
del data_test['salary']

y_train, y_test = data_train['left'], data_test['left']
lda = LinearDiscriminantAnalysis()

lda.fit(data_train, y_train)
