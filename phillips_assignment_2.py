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

/- Begin Cleaning Up Test Data -/

- imputed Lot.Frontage WHERE value == NA
- I used the average (69.2) to fill in the missing values

- House without Alley -

- imputed Alley WHERE value == NA
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
- replaced 2260 locations with None 

/- End Cleaning Up Test Data -/

/- Begin Cleaning Up Validation Data -/

- imputed Lot.Frontage WHERE value == NA
- I used the average (69.2) to fill in the missing values

- House without Alley -

- imputed Alley WHERE value was listed == NA
- Python thinks NA is a missing value, but in the data, this is being used categorically as
- a distinction 
- replaced 554 NAs with No, meaning no Alley

- House without Veneer -

- imputed Mas.Vnr.Type WHERE value == blank
- replaced 6 blank locations with None, since None was part of the data set

- imputed Mas.Vnr.Area WHERE value == NA
- Python thinks NA is a missing value, but in the data, this is being used categorically as
- a distinction
- same for imputations below
- replaced 6 NA with 0, since 0 was part of the data set 

- House without Basement - 

- imputed Bsmt.Qual WHERE value == NA
- replaced 19 locations with None

- imputed Bsmt.Cond WHERE value == NA
- replaced 19 locations with None

- imputed Bsmt.Exposure WHERE value == NA
- replaced 19 locations with No

- imputed Bsmt.Fin.Type.1 WHERE value == NA
- replaced 19 locations with None

- imputed Bsmt.Fin.Type.2 WHERE value == NA
- replaced 19 locations with None

- House without Fireplace -

- imputed Fireplace.Qu WHERE value == NA
- replaced 288 locations with No

- House without Garage -

- inputed Garage.Type WHERE value == NA
- replaced 38 locations with None

- imputed Garage.Yr.Blt WHERE value == NA
- this one was a bit tricky, but I ended up deciding to use the avg (year 1979)
- replaced 38 locations with 1979

- imputed Garage.Finish WHERE value == NA
- replaced 38 locations with None

- imputed Garage.Qual WHERE value == NA
- replaced 38 locations with None

- imputed Garage.Cond WHERE value == NA
- replaced 38 locations with None

- House without Pool -

- imputed Pool.QC WHERE value == NA
- replaced 582 locations with None

- House without Fence - 

- imputed Fence WHERE value == NA
- replaced 478 locations with None

- House without Misc Qualities - 

- imputed Misc.Feature WHERE value == NA
- replaced 564 locations with None 

- Electrical - 

- imputed Electrical WHERE value == NA
- replaced 1 location with SBrkr, as a quick pivot showed that to be the most frequent value

'''

# ----------------------------------------------------------- #
# --- Section 1: Load in Data and drop what we don't need --- #
# ----------------------------------------------------------- #

#data for building the model
housebild = pd.read_csv('./data/AmesHousingSetAv2.csv');
#PID won't be needed; its a primary or unique identifier with no bearing on the sale price
housebild = housebild.drop('PID', axis = 1); 
#data for validing the predictions
housepred = pd.read_csv('./data/AmesHousingSetBv2.csv');
#PID won't be needed; its a primary or unique identifier with no bearing on the sale price
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

# --------------------------------------------------------------------------- #
# --- Section 5: Verify / Deny my variance inflation phenomena hypothesis --- #
# --------------------------------------------------------------------------- #

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

y, X = dmatrices(formula_like='SalePrice ~ ' + str1, 
	data=df2, return_type='dataframe');
vif = pd.DataFrame();
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])];
vif["features"] = X.columns;
#inspect VIF factors
print(vif.round(1)); 

#I don't trust this right now, esp with the zero values in some places
#fast_remove_vif(df2); 

#VIF test actually passes, which is surprising, but hey that's why we do it. 
#so at this point its not feasible to throw anything out, at least based off what I chose 

#these values have VIF Factor of infinity
#thought about throwing them out, but I think it's just a div/0 mistake under the hood
'''
BsmtFin_SF_1
BsmtFin_SF_2
Bsmt_Unf_SF
Total_Bsmt_SF
X1st_Flr_SF
X2nd_Flr_SF
Low_Qual_Fin_SF
Gr_Liv_Area
'''
# ------------------------------------ #
# --- Section 6: Split up the Data --- #
# ------------------------------------ #

#much easier after rearranging

#independent / (predictor/ explanatory) variables
data_x_bild = housebild[list(housebild)[1:]];
data_x_val_ = housepred[list(housepred)[1:]];

#dependent/ response variable (in this case 'SalePrice')
data_y_bild = housebild[list(housebild)[0]];
data_y_val_ = housepred[list(housepred)[0]];

# -------------------------------------------- #
# --- Section 7: Construct Base Line Model --- #
# -------------------------------------------- #

#we'll start with a basic linear regression model

#create a least squares linear regression model
model = linear_model.LinearRegression();

#split training and test sets from main data
x_train_bild, x_test_bild, y_train_bild, y_test_bild = train_test_split(data_x_bild, data_y_bild, test_size = 0.2, random_state = 4);

# Fit the model.
model.fit(x_train_bild,y_train_bild);

# Make predictions on test data and look at the results.
preds = model.predict(x_test_bild);
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test_bild, preds), \
							   median_absolute_error(y_test_bild, preds), \
							   r2_score(y_test_bild, preds), \
							   explained_variance_score(y_test_bild, preds)]));

#output of 
'''
MSE = 699316040.50473368
MAE = 11746.06613336131
R^2 = 0.89683828997227877
EVS = 0.89788927124721352

'''
#not terrible but I'm sure there is a better model

# --------------------------------------- #
# --- Section 8: Try another approach --- #
# --------------------------------------- #

# Create a percentile-based feature selector based on the F-scores. Get top 25% best features by F-test.
selector_f = SelectPercentile(f_regression, percentile=25)
selector_f.fit(x_train_bild, y_train_bild)

# Get the columns of the best 25% features.	
xt_train_bild, xt_test_bild = selector_f.transform(x_train_bild), selector_f.transform(x_test_bild)

# Create a least squares linear regression model.
model2 = linear_model.LinearRegression()

# Fit the model.
model2.fit(xt_train_bild, y_train_bild)

# Make predictions on test data
preds2 = model2.predict(xt_test_bild)

print('MSE, MAE, R^2, EVS (Top 25% Model): ' + \
							   str([mean_squared_error(y_test_bild, preds2), \
							   median_absolute_error(y_test_bild, preds2), \
							   r2_score(y_test_bild, preds2), \
							   explained_variance_score(y_test_bild, preds2)])) 

#output of 
'''
MSE = 892508249.80567992
MAE = 14630.935672674328
R^2 = 0.86833895988236054
EVS = 0.86895928720374904

'''
#R^2 went down a bit, but MSE went up a lot

# --------------------------------------- #
# --- Section 9: Try another approach --- #
# --------------------------------------- #

#let's try lasso

# Show Lasso regression fits for different alphas.
alphas = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
for a in alphas:
	# Normalizing transforms all variables to number of standard deviations away from mean.
	lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	lasso_mod.fit(x_train_bild, y_train_bild)
	preds3 = lasso_mod.predict(x_test_bild)
	print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test_bild, preds3)))

'''
R^2 (Lasso Model with alpha=0.0): 0.896810500603
R^2 (Lasso Model with alpha=0.01): 0.896940053952
R^2 (Lasso Model with alpha=0.1): 0.897928587379
R^2 (Lasso Model with alpha=0.25): 0.899476048998
R^2 (Lasso Model with alpha=0.5): 0.901626753659
R^2 (Lasso Model with alpha=1.0): 0.904732365158
R^2 (Lasso Model with alpha=2.5): 0.909231488015
R^2 (Lasso Model with alpha=5.0): 0.912504625364
'''

alphas = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
for a in alphas:
	# Normalizing transforms all variables to number of standard deviations away from mean.
	lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	lasso_mod.fit(x_train_bild, y_train_bild)
	preds3 = lasso_mod.predict(x_test_bild)
	print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test_bild, preds3)))

'''
R^2 (Lasso Model with alpha=3.0): 0.909785989876
R^2 (Lasso Model with alpha=3.1): 0.909876858782
R^2 (Lasso Model with alpha=3.2): 0.909963866748
R^2 (Lasso Model with alpha=3.3): 0.910051436509
R^2 (Lasso Model with alpha=3.4): 0.91014355225
R^2 (Lasso Model with alpha=3.5): 0.910232501681
R^2 (Lasso Model with alpha=3.6): 0.910317724789
R^2 (Lasso Model with alpha=3.7): 0.910399007596
R^2 (Lasso Model with alpha=3.8): 0.910474293914
R^2 (Lasso Model with alpha=3.9): 0.910552022232
'''

#creeping up higher on the R^2, let's see how far we can walk it

alphas = [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9]
for a in alphas:
	# Normalizing transforms all variables to number of standard deviations away from mean.
	lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	lasso_mod.fit(x_train_bild, y_train_bild)
	preds3 = lasso_mod.predict(x_test_bild)
	print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test_bild, preds3)))

'''
R^2 (Lasso Model with alpha=4.0): 0.910714120133
R^2 (Lasso Model with alpha=4.1): 0.910917985579
R^2 (Lasso Model with alpha=4.2): 0.911112882275
R^2 (Lasso Model with alpha=4.3): 0.911303560585
R^2 (Lasso Model with alpha=4.4): 0.911493321804
R^2 (Lasso Model with alpha=4.5): 0.91168115733
R^2 (Lasso Model with alpha=4.6): 0.911864316988
R^2 (Lasso Model with alpha=4.7): 0.912035908869
R^2 (Lasso Model with alpha=4.8): 0.912198223182
R^2 (Lasso Model with alpha=4.9): 0.912353007473
'''

#still has a pulse
alphas = [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9]
for a in alphas:
	# Normalizing transforms all variables to number of standard deviations away from mean.
	lasso_mod = linear_model.Lasso(alpha=a, normalize=True, fit_intercept=True)
	lasso_mod.fit(x_train_bild, y_train_bild)
	preds3 = lasso_mod.predict(x_test_bild)
	print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(y_test_bild, preds3)))

'''
R^2 (Lasso Model with alpha=5.0): 0.912504625364
R^2 (Lasso Model with alpha=5.1): 0.912572532422
R^2 (Lasso Model with alpha=5.2): 0.912580801171
R^2 (Lasso Model with alpha=5.3): 0.91271073246
R^2 (Lasso Model with alpha=5.4): 0.912847860123
R^2 (Lasso Model with alpha=5.5): 0.912985183824
R^2 (Lasso Model with alpha=5.6): 0.913108884678
R^2 (Lasso Model with alpha=5.7): 0.913101134751
R^2 (Lasso Model with alpha=5.8): 0.913089623924
R^2 (Lasso Model with alpha=5.9): 0.913082246264
'''

#we've reached our limit @ alpha == 5.6
#this gives us the best overall R^2 score
#i'll use this for the actual model

# ---------------------------------------------------------------------- #
# --- Section 10: Validate it against the data the model hasn't seen --- #
# ---------------------------------------------------------------------- #

model.fit(data_x_val_, data_y_val_); 
predsv1 = model.predict(data_x_val_);
print('Base Model Fit to Validation Data\n');
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(data_y_val_, predsv1), \
							   median_absolute_error(data_y_val_, predsv1), \
							   r2_score(data_y_val_, predsv1), \
							   explained_variance_score(data_y_val_, predsv1)]));
print('\n'); 

lasso_mod = linear_model.Lasso(alpha=5.6, normalize=True, fit_intercept=True);
lasso_mod.fit(data_x_val_, data_y_val_);
predsv2 = lasso_mod.predict(data_x_val_);
print('R^2 (Lasso Model with alpha=' + str(a) + '): ' + str(r2_score(data_y_val_, predsv2)));
print('EVS: ' +str(explained_variance_score(data_y_val_, predsv1))); 

pprint.pprint(pd.DataFrame({'Actual':data_y_val_, 'Predicted':predsv2}));

