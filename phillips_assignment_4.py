#Assignment #4
#DSCI401B
#William (Greg) Phillips
#Working

from __future__ import division 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
data_util_file = './util/data_util.py'
import os
import sys
sys.path.append(os.path.dirname(os.path.expanduser(data_util_file))); 
import data_util as util

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
#print("Column names: "); 
#print(col_names); 

#print("\nSample data: "); 
#print(data.head(5)); 
#print(data.tail(5));

# --------------------------------------------- #
# --- Section 2: Helper function -------------- #
# --------------------------------------------- #

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
data = pd.get_dummies(data, columns=util.cat_features(data));

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
b = util.check_missing_data(data); 
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


# ---------------------------------- #
# --- Section 4: Evaluate Models --- #
# ---------------------------------- #

print("Support Vector Machine:"); 
print("%.4f" % util.accuracy(y, util.run_cv(X,y,SVC))); 
print("Random Forest:"); 
print("%.4f" % util.accuracy(y, util.run_cv(X,y,RF))); 
print("K-Nearest-Neighbors:"); 
print("%.4f" % util.accuracy(y, util.run_cv(X,y,KNN))); 

# ------------------------------------- #
# --- Section 5: Confusion Matrices --- #
# ------------------------------------- #

y = np.array(y)
class_names = np.unique(y)
np.set_printoptions(precision=2)

confusion_matrix_SVC = confusion_matrix(y, util.run_cv(X,y,SVC)); 
confusion_matrix_RF = confusion_matrix(y, util.run_cv(X,y,RF)); 
confusion_matrix_KNN = confusion_matrix(y, util.run_cv(X,y,KNN)); 

plt.figure()
util.plot_confusion_matrix(confusion_matrix_SVC, classes=class_names,
                      title='Support Vector Machine, without normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_RF, classes=class_names,
                      title='Random Forest, without normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_KNN, classes=class_names,
                      title='K-Nearest-Neighbors, without normalization')

plt.figure()
util.plot_confusion_matrix(confusion_matrix_SVC, classes=class_names, normalize=True,
                      title='Support Vector Machine, with normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_RF, classes=class_names, normalize=True,
                      title='Random Forest, with normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_KNN, classes=class_names, normalize=True,
                      title='K-Nearest-Neighbors, with normalization')

#plt.show()

# -------------------------------- #
# --- Section 6: Probabilities --- #
# -------------------------------- #

predicted_prob = util.run_prob_cv(X, y, RF, n_estimators=10); 
predicted_emp = predicted_prob[:,1]; 
is_AECOM_emp = y == 1; 
counts = pd.value_counts(predicted_emp); 
actual_prob ={}; 
for prob in counts.index:
	actual_prob[prob] = np.mean(is_AECOM_emp[predicted_emp == prob]); 
	actual_prob = pd.Series(actual_prob); 

counts = pd.concat([counts, actual_prob], axis=1).reset_index(); 
counts.columns = ['Predicted Probability', 'Count', 'Actual Probability']; 
print(counts); 