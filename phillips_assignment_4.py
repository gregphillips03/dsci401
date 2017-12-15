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
import random; 
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
from sklearn.metrics import roc_curve; 
from sklearn.metrics import auc;  
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
valdata = pd.read_excel('./data/incs_val.xlsx', sheet_name='fy16'); 
vcol_names = valdata.columns.tolist(); 
#print("Training Column names: "); 
#print(col_names); 
#print("Validation Column names: "); 
#print(vcol_names); 

#print("\nSample data: "); 
#print(data.head(5)); 
#print(data.tail(5));

#print("\nSample data: "); 
#print(valdata.head(5)); 
#print(valdata.tail(5));

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
val_target_result = valdata['Worker Type']; 
#let's alter the values, so that everytime that we see:
#AECOM Employee, we set it to 1
#Everything else (contractor) set it to 0
y = np.where(target_result == 'AECOM Employee', 1, 0);
valy = np.where(val_target_result == 'AECOM Employee', 1, 0);

#if I had anything additional to drop, i'd specify it here
#but I get rid of most of this outside the working environment
to_drop =[];
data = data.drop(['Worker Type'], axis=1); 
valdata = valdata.drop(['Worker Type'], axis=1); 

#transform using a label encoder
data = pd.get_dummies(data, columns=util.cat_features(data));
valdata = pd.get_dummies(valdata, columns=util.cat_features(valdata)); 
 
feature_space = data; 
val_feature_space = valdata; 

#remove feature space in case we need it later
features = feature_space.columns;  
valfeatures = val_feature_space.columns; 
X = feature_space.as_matrix().astype(np.float); 
valX = val_feature_space.as_matrix().astype(np.float); 

#apply a scaler to the predictors
scaler = StandardScaler(); 
X = scaler.fit_transform(X);
valX = scaler.fit_transform(valX); 

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

b = util.check_missing_data(valdata); 
if(b):
	print('Found Missing Data'); 
	show_name(valdata); 
	print('\n');
else:
	print('No Missing Data!');
	show_name(valdata); 
	print('\n');

#check to make sure that we've not done anything crazy at this point
print("Training Feature space contains %d records and %d columns" % X.shape); 
print("Number of Response Types:", np.unique(y)); 

print("Validation Feature space contains %d records and %d columns" % valX.shape); 
print("Number of Response Types:", np.unique(valy)); 


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
clf4 = SVC(probability=True); 

scoring = {'F1': 'f1'}; 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4);

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
traindf = pd.DataFrame(pd.DataFrame({'Actual': y_test, 'Predicted': preds}));
writer = pd.ExcelWriter('./gen/trainoutput.xlsx'); 
traindf.to_excel(writer, 'Sheet1'); 
writer.save(); 

# -------------------------------- #
# --- Section 7: Validation ------ #
# -------------------------------- #

#let's see how the model performs on a data set it hasn't seen. 
valpreds = best_voting_mod.predict(valX); 
print('Voting Ensemble Model Real Score: ' + str(best_voting_mod.score(valX, valy))); 
valdf = pd.DataFrame(pd.DataFrame({'Actual': valy, 'Predicted': valpreds}));
valwriter = pd.ExcelWriter('./gen/validoutput.xlsx');  
valdf.to_excel(valwriter, 'Sheet1'); 
valwriter.save(); 

class_names=np.unique(valy); 
confusion_matrix_ensemble = confusion_matrix(valy, valpreds);
plt.figure()
util.plot_confusion_matrix(confusion_matrix_ensemble, classes=class_names,
                      title='SVM, KNN, RF Ensemble, without normalization')
#plt.show(); 

false_positive_rate, true_positive_rate, thresholds = roc_curve(valy, valpreds); 
roc_auc = auc(false_positive_rate, true_positive_rate); 
plt.title('Receiver Operating Characteristic'); 
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc); 
plt.legend(loc='lower right'); 
plt.plot([0,1], [0,1], 'r--'); 
plt.xlim([-0.1, 1.2]); 
plt.ylim([-0.1, 1.2]); 
plt.ylabel('True Positive Rate'); 
plt.xlabel('False Positive Rate'); 
#plt.show(); 

# -------------------------------- #
# --- Section 8: Probabilities --- #
# -------------------------------- #

voting_mod = VotingClassifier(estimators=[('svm', clf4), ('rf', clf2), ('knn', clf3)], voting='soft');
param_grid = {'svm__C':[0.2, 0.5, 1.0, 2.0, 5.0, 10.0], 'rf__n_estimators':[5, 10, 50, 100], 'rf__max_depth': [3, 6, None]};
best_voting_mod = GridSearchCV(estimator=voting_mod, param_grid=param_grid, cv=5, scoring=scoring, refit='F1'); 
best_voting_mod.fit(x_train, y_train);
valprobs = best_voting_mod.predict_proba(valX); 
prob_pos = valprobs.transpose()[1]; 
prob_neg = valprobs.transpose()[0]; 

pred_df = pd.DataFrame({'Actual': valy, 'Predicted Class': valpreds, 'P(1)': prob_pos, 'P(0)': prob_neg});
probwriter = pd.ExcelWriter('./gen/proboutput.xlsx'); 
pred_df.to_excel(probwriter, 'Sheet1');  
probwriter.save(); 

