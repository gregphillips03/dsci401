#Assignment #4
#DSCI401B
#William (Greg) Phillips
#Working

from __future__ import division 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

# -------------------------------------- #
# --- Section 0: Meta Data & Caveats --- #
# -------------------------------------- #

'''
The following transformations were made to the data outside of the work environment:

- The data set contained the column 'Date of Incident', which was renamed to 'Month'
- Each record contained the specific calendar day which the incident occurred. These fields were mapped
to the corresponding month of the year, as a text label; i.e., 03/04/2017 became 'March'. 

- Changed 'Worker Type' WHERE value == 'Contractor/Subcontractor (Managed)'
- Replaced 1040 locations with 'Contractor'

- Changed 'Worker Type' WHERE value == 'Contractor/Subcontractor (Unmanaged)'
- Replaced 1316 locations with 'Contractor'

The following records were removed from the data set:

- The data set contains more than 'AECOM Employees' and 'Contractors'. It also contains joint venture
partners, as well as members of the public. 

- Removed records WHERE value of 'Worker Type' == 'JV Partner'. 
- Removed a total of 6 records. 

- Removed records WHERE value of 'Worker Type' == 'Third Party/Public'. 
- Removed a total of 38 records. 

'''

# --------------------------------------------- #
# --- Section 1: Importation and First Look --- #
# --------------------------------------------- #

data = pd.read_excel('./data/incs.xlsx', sheet_name='fy17');
col_names = data.columns.tolist(); 
print("Column names: "); 
print(col_names); 

print("\nSample data: "); 
print(data.head(5)); 
print(data.tail(5));

# ------------------------------------------------ #
# --- Section 2: Define some utility functions --- #
# ------------------------------------------------ #

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

#Function shows the DataFrame name
def show_name(df):
	name = [x for x in globals() if globals()[x] is df][0]; 
	print("DataFrame Name is: %s" %name); 

# ------------------------------------------------- #
# --- Section 3: Transformation and Cleaning up --- #
# ------------------------------------------------- #

#isolate our target data
target_result = data['Worker Type']; 
#let's alter the values, so that everytime that we see:
#AECOM Employee, we set it to 1
#Everything else (contractor) set it to 0
y = np.where(target_result == 'AECOM Employee', 1, 0)

#transform using a label encoder
data = pd.get_dummies(data, columns=cat_features(data));

#if I had anything to drop, i'd specify it here
#but I get rid of this outside the working environment
to_drop =[]; 
feature_space = data.drop(to_drop, axis=1);

#remove feature space in case we need it later
features = feature_space.columns;  
X = feature_space.as_matrix().astype(np.float); 

#apply a scaler to the predictors
scaler = StandardScaler(); 
X = scaler.fit_transform(X);

#let's check to see if there's missing data
b = check_missing_data(data); 
if(b):
	print('Found Missing Data'); 
	show_name(data); 
	print('\n');
else:
	print('No Missing Data!');
	show_name(data); 
	print('\n');

#check to make sure that we've not done anything crazy at this point
print("Feature space contains %d records and %d columns" % X.shape); 
print("Number of Response Types:", np.unique(y));  

