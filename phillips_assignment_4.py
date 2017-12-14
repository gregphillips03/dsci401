#Assignment #4
#DSCI401B
#William (Greg) Phillips
#Working

from __future__ import division;
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import itertools;
import pprint; 
from sklearn.preprocessing import StandardScaler;
from sklearn.cross_validation import KFold;
from sklearn.svm import SVC;
from sklearn.ensemble import RandomForestClassifier as RF;
from sklearn.ensemble import VotingClassifier
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier as KNN;
from sklearn.naive_bayes import BernoulliNB as BNB; 
from sklearn.naive_bayes import GaussianNB as GNB; 
from sklearn.tree import DecisionTreeClassifier as DTC; 
from sklearn.metrics import confusion_matrix;
from sklearn.metrics import recall_score;
from sklearn.metrics import precision_score;  
from sklearn.metrics import f1_score; 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

data_util_file = './util/data_util.py';
import os;
import sys;
sys.path.append(os.path.dirname(os.path.expanduser(data_util_file))); 
import data_util as util;

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

#if I had anything additional to drop, i'd specify it here
#but I get rid of most of this outside the working environment
to_drop =[];
data = data.drop(['Worker Type'], axis=1); 

#transform using a label encoder
data = pd.get_dummies(data, columns=util.cat_features(data));
 
feature_space = data; 

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
	#print('No Missing Data!');
	#show_name(data); 
	print('\n');

#check to make sure that we've not done anything crazy at this point
#print("Feature space contains %d records and %d columns" % X.shape); 
#print("Number of Response Types:", np.unique(y));  


# ---------------------------------- #
# --- Section 4: Evaluate Models --- #
# ---------------------------------- #

# print("Support Vector Machine:"); 
# print("%.4f" % util.accuracy(y, util.run_cv(X,y,SVC))); 
# print("Recall"); 
# print("%.4f" % recall_score(y, util.run_cv(X,y,SVC)));
# print("Precision"); 
# print("%.4f" % precision_score(y, util.run_cv(X,y,SVC)));
# print("F1"); 
# print("%.4f" % f1_score(y, util.run_cv(X,y,SVC)));

# print("Random Forest:"); 
# print("%.4f" % util.accuracy(y, util.run_cv(X,y,RF))); 
# print("Recall"); 
# print("%.4f" % recall_score(y, util.run_cv(X,y,RF)));
# print("Precision"); 
# print("%.4f" % precision_score(y, util.run_cv(X,y,RF)));
# print("F1"); 
# print("%.4f" % f1_score(y, util.run_cv(X,y,RF)));

# print("K-Nearest-Neighbors:"); 
# print("%.4f" % util.accuracy(y, util.run_cv(X,y,KNN)));
# print("Recall"); 
# print("%.4f" % recall_score(y, util.run_cv(X,y,KNN))); 
# print("Precision"); 
# print("%.4f" % precision_score(y, util.run_cv(X,y,KNN)));
# print("F1"); 
# print("%.4f" % f1_score(y, util.run_cv(X,y,KNN)));

# print("Naive Bayes Bernoulli:"); 
# print("%.4f" % util.accuracy(y, util.run_cv(X,y,BNB)));
# print("Recall"); 
# print("%.4f" % recall_score(y, util.run_cv(X,y,BNB)));
# print("Precision"); 
# print("%.4f" % precision_score(y, util.run_cv(X,y,BNB)));
# print("F1"); 
# print("%.4f" % f1_score(y, util.run_cv(X,y,BNB)));

# print("Naive Bayes Gaussian:"); 
# print("%.4f" % util.accuracy(y, util.run_cv(X,y,GNB)));
# print("Recall"); 
# print("%.4f" % recall_score(y, util.run_cv(X,y,GNB)));
# print("Precision"); 
# print("%.4f" % precision_score(y, util.run_cv(X,y,GNB)));
# print("F1"); 
# print("%.4f" % f1_score(y, util.run_cv(X,y,GNB)));

# print("Decision Tree (Gini Impurity):"); 
# print("%.4f" % util.accuracy(y, util.run_cv(X,y,DTC)));
# print("Recall"); 
# print("%.4f" % recall_score(y, util.run_cv(X,y,DTC)));
# print("Precision"); 
# print("%.4f" % precision_score(y, util.run_cv(X,y,DTC)));
# print("F1"); 
# print("%.4f" % f1_score(y, util.run_cv(X,y,DTC)));


# ------------------------------------- #
# --- Section 5: Confusion Matrices --- #
# ------------------------------------- #

y = np.array(y)
class_names = np.unique(y)
np.set_printoptions(precision=2)

confusion_matrix_SVC = confusion_matrix(y, util.run_cv(X,y,SVC)); 
confusion_matrix_RF = confusion_matrix(y, util.run_cv(X,y,RF)); 
confusion_matrix_KNN = confusion_matrix(y, util.run_cv(X,y,KNN)); 
confusion_matrix_BNB = confusion_matrix(y, util.run_cv(X,y,BNB)); 
confusion_matrix_GNB = confusion_matrix(y, util.run_cv(X,y,GNB)); 
confusion_matrix_DTC = confusion_matrix(y, util.run_cv(X,y,DTC)); 

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
util.plot_confusion_matrix(confusion_matrix_BNB, classes=class_names,
                      title='Naive Bayes Bernoulli, without normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_GNB, classes=class_names,
                      title='Naive Bayes Gaussian, without normalization')
plt.figure()
util.plot_confusion_matrix(confusion_matrix_DTC, classes=class_names,
                      title='Decision Tree (Gini Impurity), without normalization')

#plt.show()

# -------------------------------- #
# --- Section 6: Voting ---------- #
# -------------------------------- #

clf1 = SVC(); 
clf2 = RF(); 
clf3 = KNN(); 

scoring = {'F1': 'f1'}; 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

voting_mod = VotingClassifier(estimators=[('svm', clf1), ('rf', clf2), ('knn', clf3)], voting='hard');

# Set up params for combined Grid Search on the voting model. 
#Notice the convention for specifying parameters foreach of the different models.
param_grid = {'svm__C':[0.2, 0.5, 1.0, 2.0, 5.0, 10.0], 'rf__n_estimators':[5, 10, 50, 100], 'rf__max_depth': [3, 6, None]}; 

best_voting_mod = GridSearchCV(estimator=voting_mod, param_grid=param_grid, cv=5, scoring=scoring, refit='F1'); 
best_voting_mod.fit(x_train, y_train);
preds = best_voting_mod.predict(x_test); 
print('Voting Ensemble Model Test Score: ' + str(best_voting_mod.score(x_test, y_test)));
#spit it out to the commandline  
pprint.pprint(pd.DataFrame({'Actual': y_test, 'Predicted': preds})); 

#spit out the file to something human readable
outdf = pd.DataFrame(pd.DataFrame({'Actual': y_test, 'Predicted': preds}));
writer = pd.ExcelWriter('./gen/output.xlsx'); 
outdf.to_excel(writer, 'Sheet1'); 
writer.save(); 

# -------------------------------- #
# --- Section 7: Probabilities --- #
# -------------------------------- #

# predicted_prob = util.run_prob_cv(X, y, best_voting_mod, n_estimators=10); 
# predicted_emp = predicted_prob[:,1]; 
# is_AECOM_emp = y == 1; 
# counts = pd.value_counts(predicted_emp); 
# actual_prob ={}; 
# for prob in counts.index:
# 	actual_prob[prob] = np.mean(is_AECOM_emp[predicted_emp == prob]); 
# 	actual_prob = pd.Series(actual_prob); 

# counts = pd.concat([counts, actual_prob], axis=1).reset_index(); 
# counts.columns = ['Predicted Probability', 'Count', 'Actual Probability']; 
# print(counts); 