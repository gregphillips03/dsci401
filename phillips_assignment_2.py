#assignment #2
#dsci401B
#william (greg) phillips

import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

housebild = pd.read_csv('./data/AmesHousingSetA.csv');
housepred = pd.read_csv('./data/AmesHousingSetB.csv');