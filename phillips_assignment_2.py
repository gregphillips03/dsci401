#assignment #2
#dsci401B
#william (greg) phillips

import pandas as pd
import matplotlib.pyplot as plt
import pprint
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import preprocessing

# ------------- #
# --- Notes --- #
# ------------- #

'''I found that cleanint up the data (imagine that) a bit difficult in Python
versus how we'd do it in R. I did the following outside of this script by hand

- imputed Lot.Frontage WHERE value was listed as NA
- I used the average (69.2) to fill in the missing values 
df_vals = df_vals[~df_vals['somecolname'].isnull()] '''

# ----------------------------------------------------------- #
# --- Section 0: Load in Data and drop what we don't need --- #
# ----------------------------------------------------------- #

#data for building the model
housebild = pd.read_csv('./data/AmesHousingSetA.csv');
#PID won't be needed
housebild = housebild.drop('PID', axis = 1); 
#data for validing the predictions
housepred = pd.read_csv('./data/AmesHousingSetB.csv');
#PID won't be needed
housepred = housepred.drop('PID', axis = 1); 

# ----------------------------------------------- #
# --- Section 1: Define some helper functions --- #
# ----------------------------------------------- #

# Get a list of the categorical features for a given dataframe.
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe.
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected);

#Function plots a graphical correlation matrix for each pair of columns in the dataframe.
#Input: ((df: pandas DataFrame)(size: vertical and horizontal size of the plot))
def plot_corr(df,size=10): 
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);

#Function pulls out each column where there is a missing value
#returns a list of column names
def missing_cols(df):
	a = [col for col in df.columns if df[col].isnull().any()]
	return a;

#Function checks to see if there are missing values within the dataframe
#If true, prints number of missing values, then calls missing_cols to 
#list rows with missing values
def check_missing_data(df):
	b = df.isnull().any().any();
	if(b):
		print('No of missing vals: ' + str(df.isnull().sum().sum()));
		a = missing_cols(df); 
		print('Cols without values: ' + str(a)); 

#Function moves specified column to a specified index
def move_to_index(df, colName, index=0):
	cols = list(df); 
	cols.insert(index, cols.pop(cols.index(colName)));
	df = df.ix[:, cols]; 
	return df; 

# -------------------------------------- #
# --- Section 2: Data transformation --- #
# -------------------------------------- #

#let's check to see if there's missing data
check_missing_data(housebild); 
check_missing_data(housepred);  

#move SalePrice to the 0 index in housebild
#becaue I'm picky about locations
housebild = move_to_index(housebild, 'SalePrice'); 

#let's do the same for the validation data
housepred = move_to_index(housepred, 'SalePrice'); 

#transform the df to a one-hot encoding
housebild = pd.get_dummies(housebild, columns=cat_features(housebild));
housepred = pd.get_dummies(housepred, columns=cat_features(housepred));

#surpirse, there's missing data (as if we thought we'd get away with that)
#using most frequent as the string, since I can't find a similar 'replace with this' as in R
#imp = preprocessing.Imputer(missing_values='NaN', strategy='most_frequent', axis=0);
#housebild = imp.fit_transform(housebild);


# ------------------------------------------------------------------------------------------ #
# --- Section 3: Do some exploratory analysis on the data to visually find relationships --- #
# ------------------------------------------------------------------------------------------ #

#make a data frame to play around with
df1 = housebild.filter(['SalePrice', 'Year.Built', 'Lot.Area', 'Year.Remod.Add', 'Overall.Cond', 
	'Lot.Frontage', 'Overall.Qual', 'Mas.Vnr.Area', 'Total.Bsmt.SF', 'Gr.Liv.Area'], axis =1);

#let's use seaborn's heatmap, hopefully it looks simliar to what we can do in R
corr = df1.corr(); 
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
#ok that's much easier to understand than matshow and a bit sexier as well
plt.show(); 

# --- Section 4: --- #
