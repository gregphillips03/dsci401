#Assignment #2
#DSCI401B
#William (Greg) Phillips

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pprint
import seaborn as sns
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# -------------------------------------- #
# --- Section 0: Meta Data & Caveats --- #
# -------------------------------------- #

'''I found that cleaning up the data (imagine that) a bit difficult in Python
versus how we'd do it in R. I did the following outside of this script by hand

- imputed Lot.Frontage WHERE value was listed as NA
- I used the average (69.2) to fill in the missing values

- House without Alley -

- imputed Alley WHERE value was listed as NA
- Python thinks NA is a missing value, but in the data, this is being used categorically as
- a distinction 
- replaced NAs with No, meaning no Alley

- House without Veneer -

- imputed Mas.Vnr.Type WHERE value == blank
- replaced 17 blank locations with None, since None was part of the data set

- imputed Mas.Vnr.Area WHERE value == NA
- Python thinks NA is a missing value, but in the data, this is being used categorically as
- a distinction
- same for imputations below
- replaced 17 NA with 0, since 0 was part of the data set 

- House without Basement - 

- imputed Bsmt.Qual WHERE value == blank
- replaced 1 location with TA as a quick pivot table revealed that to be the most frequent value
- without an ontolgoy or data dictionary its hard to tell what to use, ergo most frequent

- imputed Bsmt.Qual WHERE value == NA
- replaced 60 locations with None

- imputed Bsmt.Cond WHERE value == blank
- replaced 1 location with TA as a quick pivot table revealed that to be the most frequent value
- without an ontolgoy or data dictionary its hard to tell what to use, ergo most frequent

- imputed Bsmt.Cond WHERE value == NA
- replaced 60 locations with None

- imputed Bsmt.Exposure WHERE value == blank
- replaced 4 locations with No

- imputed Bsmt.Exposure WHERE value == NA
- replaced 60 locations with No

- imputed Bsmt.Fin.Type.1 WHERE value == NA
- replaced 60 locations with None

- imputed Bsmt.Fin.Type.1 WHERE value == blank
- replaced 1 location with None

- imputed Bsmt.Fin.SF.1 WHERE value == NA
- replaced 1 location with 0

- imputed Bsmt.Fin.Type.2 WHERE value == NA
- replaced 60 locations with None

- imputed Bsmt.Fin.Type.2 WHERE value == blank
- replaced 2 locations with None

- imputed Bsmt.Fin.SF.2 WHERE value == NA
- replaced 1 location with 0

- imputed Bsmt.Unf.SF WHERE value == NA
- replaced 1 location with 0

- imputed Total.Bsmt.SF WHERE value == NA
- replaced 1 location with 0

- imputed Bsmt.Full.Bath WHERE value == NA
- replaced 2 locations with 0

- imputed Bsmt.Half.Bath WHERE value == NA
- replaced 2 locations with 0

- House without Fireplace -

- imputed Fireplace.Qu WHERE value == NA
- replaced 1134 locations with No

- House without Garage -

- inputed Garage.Type WHERE value == NA
- replaced 119 locations with None

- imputed Garage.Yr.Blt WHERE value == NA
- this one was a bit tricky, but I ended up deciding to use the avg (year 1977)
- replaced 121 locations with 1977

- imputed Garage.Yr.Blt WHERE value == 2207
- considered this as a data entry error
- replaced 1 location with 2007

- imputed Garage.Finish WHERE value == NA
- replaced 119 locations with None

- imputed Garage.Finish WHERE value == blank
- replaced 2 locations with None

- imputed Garage.Cars WHERE value == NA
- replaced 1 location with 0

- imputed Garage.Area WHERE value == NA
- replaced 1 location with 0

- imputed Garage.Qual WHERE value == blank
- replaced 1 locatoin with TA as a quick pivot table revealed that to be the most frequent value
- without an ontolgoy or data dictionary its hard to tell what to use, ergo most frequent

- imputed Garage.Qual WHERE value == NA
- replaced 120 locations with None

- imputed Garage.Cond WHERE value == blank
- replaced 1 location with TA as a quick pivot table revealed that to be the most frequent value
- without an ontolgoy or data dictionary its hard to tell what to use, ergo most frequent

- imputed Garage.Cond WHERE value == NA
- replaced 120 locations with None

- House without Pool -

- imputed Pool.QC WHERE value == NA
- replaced 2335 locations with None

- House without Fence - 

- imputed Fence WHERE value == NA
- replaced 1880 locations with None

- House without Misc Qualities - 

- imputed Misc.Feature WHERE value == NA
- replaced 2260 locations with None '''

# ----------------------------------------------------------- #
# --- Section 1: Load in Data and drop what we don't need --- #
# ----------------------------------------------------------- #

#data for building the model
housebild = pd.read_csv('./data/AmesHousingSetAv2.csv');
#PID won't be needed, its a primary or unique identifier with no bearing on the sale price
housebild = housebild.drop('PID', axis = 1); 
#data for validing the predictions
housepred = pd.read_csv('./data/AmesHousingSetB.csv');
#PID won't be needed its a primary or unique identifier with no bearing on the sale price
housepred = housepred.drop('PID', axis = 1); 

# ----------------------------------------------- #
# --- Section 2: Define some utility functions --- #
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
	return b;

#Function moves specified column to a specified index
def move_to_index(df, colName, index=0):
	cols = list(df); 
	cols.insert(index, cols.pop(cols.index(colName)));
	df = df.ix[:, cols]; 
	return df; 

#Function removes records where there is a missing value
def drop_records_with_missing_vals(df, colName):
	df = df[~df['colName'].isnull()]; 
	return df;

#Function looks for periods in column names and swaps them out for underscores
def drop_periods(df):
	df.columns = [c.replace('.', '_') for c in df.columns]; 
	print('Dropped periods');
	show_name(df); 
	print('\n'); 

#calculate variance inflation and get rid of those that are too influential
#X = pandas data frame
def fast_remove_vif(X):
    thresh = 5.0
    variables = range(X.shape[1])

    for i in np.arange(0, len(variables)):
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]
        print(vif)
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]

    print('Remaining variables:')
    print(X.columns[variables])
    return X

#Function shows the DataFrame name
def show_name(df):
	name = [x for x in globals() if globals()[x] is df][0]; 
	print("DataFrame Name is: %s" %name); 

# -------------------------------------- #
# --- Section 3: Data transformation --- #
# -------------------------------------- #

#move SalePrice to the 0 index in housebild
#becaue I'm picky about locations
housebild = move_to_index(housebild, 'SalePrice'); 
#replace periods with underscores
drop_periods(housebild); 

#let's do the same for the validation data
housepred = move_to_index(housepred, 'SalePrice');
#replace periods with underscores
drop_periods(housepred); 

#let's check to see if there's missing data
b1 = check_missing_data(housebild); 
if(b1):
	print('Found Missing Data'); 
	show_name(housebild); 
	print('\n');
else:
	print('No Missing Data!');
	show_name(housebild); 
	print('\n');

b2 = check_missing_data(housepred); 
if(b2):
	print('Found Missing Data');
	show_name(housepred);  
	print('\n');
else:
	print('No Missing Data!');
	show_name(housepred);
	print('\n');  

#transform the df to a one-hot encoding
housebild = pd.get_dummies(housebild, columns=cat_features(housebild));
housepred = pd.get_dummies(housepred, columns=cat_features(housepred));

# ------------------------------------------------------------------------------------------ #
# --- Section 4: Do some exploratory analysis on the data to visually find relationships --- #
# ------------------------------------------------------------------------------------------ #

#make a data frame to play around with
#based off attributes I *think* will have a large amount of influence on the dependent variable
df1 = housebild.filter(['SalePrice', 'Year_Built', 'Lot_Area', 'Year_Remod_Add', 'Overall_Cond', 
	'Lot_Frontage', 'Overall_Qual', 'Mas.Vnr_Area', 'Total_Bsmt_SF', 'Gr_Liv_Area'], axis =1);

#let's use seaborn's heatmap, hopefully it looks simliar to what we can do in R
corr = df1.corr(); 
heat = sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values);
#ok that's much easier to understand than matshow and a bit sexier as well
plt.show(heat); 

'''It appears as if there are going to be predictors that influence other predictors. what
jumps out at first is the overall quality vand the year remodel was added, which are probably
supportive of each other, so multicollinearity is probably happening, so 
lasso is probably the best approach here '''

# -------------------------------------------------------------------- #
# --- Section 5: Verify my variance inflation phenomena hypothesis --- #
# -------------------------------------------------------------------- #

#bring in a fresh copy. the getdummies function causes issues when parsing under the hood?
#or is it having trouble parsing periods? 
df2 = pd.read_csv('./data/AmesHousingSetAv2.csv');
df2 = df2.drop('PID', axis = 1); 
#drop out missing vals 
df2.dropna(); 
#drop non-numeric cols
df2 = df2._get_numeric_data();
drop_periods(df2); 

#this causes problems when sent over to dmatrices for some reason
features = list(df2); 
features.remove("SalePrice"); 
str1 = " + ".join(features); 
#this causes problems when sent over to dmatrices for some reason

#hand jambing for now
str2 = "Year_Built + Lot_Area + Year_Remod_Add + Overall_Cond + Lot_Frontage + Overall_Qual + Mas_Vnr_Area + Total_Bsmt_SF + Gr_Liv_Area"

#break into left and right hand side; y and X
#get y and X dataframes based on this regression:
#so sales price 'is dependent up on all the features in the string'

y, X = dmatrices(formula_like='SalePrice ~ ' + str2, 
	data=df2, return_type='dataframe');
vif = pd.DataFrame();
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])];
vif["features"] = X.columns;
#inspect VIF factors
print(vif.round(1)); 
#fast_remove_vif(df2); 

#VIF test actually passes, which is surprising, but hey that's why we do it. 
#so at this point its not feasible to throw anything out, at least based off what I chose 

# ------------------------------------ #
# --- Section 6: Split up the Data --- #
# ------------------------------------ #

#much easier after rearranging

#independent / (predictor/ explanatory) variables
data_x = housebild[list(housebild)[1:]];

#dependent/ response variable (in this case 'SalePrice')
data_y = housebild[list(housebild)[0]];

# -------------------------------------------- #
# --- Section 7: Construct Base Line Model --- #
# -------------------------------------------- #


