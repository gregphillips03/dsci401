#Assignment #3
#DSCI401B
#William (Greg) Phillips
#Submission

import pandas as pd
import pprint
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import ensemble 
from sklearn.metrics import classification_report

# -------------------------------------- #
# --- Section 0: Meta Data & Caveats --- #
# -------------------------------------- #

'''
Outside of the work environment, I change the values of the 'Churn' field
Yes to 1
No to 0
'''

# ----------------------------------------------------------- #
# --- Section 1: Load in Data and drop what we don't need --- #
# ----------------------------------------------------------- #

#data for building the model
churn_data = pd.read_csv('./data/churn_data.csv');
#CustID won't be needed; its a primary or unique identifier with no bearing on the outcome
churn_data = churn_data.drop('CustID', axis = 1); 
#data for validing the predictions
churn_vald = pd.read_csv('./data/churn_validation.csv');
#CustID won't be needed; its a primary or unique identifier with no bearing on the soutcome
churn_vald = churn_vald.drop('CustID', axis = 1); 

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

# Print out common error metrics for the binary classifications.
def print_multiclass_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Avg. F1 (Micro): ' + str(f1_score(y_test, preds, average='micro')))
	print('Avg. F1 (Macro): ' + str(f1_score(y_test, preds, average='macro')))
	print('Avg. F1 (Weighted): ' + str(f1_score(y_test, preds, average='weighted')))
	print(classification_report(y_test, preds))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))

# -------------------------------------- #
# --- Section 3: Data transformation --- #
# -------------------------------------- #

#move Churn to the 0 index in housebild
#becaue I'm picky about locations
churn_data = move_to_index(churn_data, 'Churn'); 
#let's do the same for the validation data
churn_vald = move_to_index(churn_vald, 'Churn'); 

#let's check to see if there's missing data
#take a look at the test data
b1 = check_missing_data(churn_data); 
if(b1):
	print('Found Missing Data'); 
	show_name(churn_data); 
	print('\n');
else:
	print('No Missing Data!');
	show_name(churn_data); 
	print('\n');

#take a look at the validation data
b2 = check_missing_data(churn_vald); 
if(b2):
	print('Found Missing Data');
	show_name(churn_vald);  
	print('\n');
else:
	print('No Missing Data!');
	show_name(churn_vald);
	print('\n'); 

#transform the dfs to a one-hot encoding
churn_data = pd.get_dummies(churn_data, columns=cat_features(churn_data));
churn_vald = pd.get_dummies(churn_vald, columns=cat_features(churn_vald));

# ------------------------------------ #
# --- Section 4: Split up the Data --- #
# ------------------------------------ #

#much easier after rearranging
scaler = StandardScaler()

#independent / (predictor/ explanatory) variables
churn_data_x = churn_data[list(churn_data)[1:]];
churn_data_x = scaler.fit_transform(churn_data_x);
churn_vald_x = churn_vald[list(churn_vald)[1:]];
churn_vald_x = scaler.fit_transform(churn_vald_x); 

#dependent/ response variable (in this case 'Churn')
churn_data_y = churn_data[list(churn_data)[0]];
churn_vald_y = churn_vald[list(churn_vald)[0]];

#split training and test sets from main data
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(churn_data_x, churn_data_y, 
	test_size = 0.2, random_state = 4);

#split training and test sets from main data
x_train_vald, x_test_vald, y_train_vald, y_test_vald = train_test_split(churn_vald_x, churn_vald_y, 
	test_size = 0.2, random_state = 4); 

# --------------------------------------- #
# --- Section 5: K-Nearest Evaluation --- #
# --------------------------------------- #

# Build a sequence of models for k = 2, 4, 6, 8, ..., 20.
ks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
for k in ks:
	# Create model and fit.
	kmod = neighbors.KNeighborsClassifier(n_neighbors=k);
	kmod.fit(x_train_data, y_train_data);
	# Make predictions - both class labels and predicted probabilities.
	preds_data = kmod.predict(x_test_data);
	print('---------- EVALUATING MODEL (Data): k = ' + str(k) + ' -------------------');
	# Look at results.
	print('Accuracy: ' + str(accuracy_score(y_test_data, preds_data)));
	print('Precison: ' + str(precision_score(y_test_data, preds_data)));
	print('Recall: ' + str(recall_score(y_test_data, preds_data)));
	print('F1: ' + str(f1_score(y_test_data, preds_data)));
	print('ROC AUC: ' + str(roc_auc_score(y_test_data, preds_data)));
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test_data, preds_data)));

# --------------------------------------------------------- #
# --- Section 6: 2-Class Logistic Regression Evaluation --- #
# --------------------------------------------------------- #

# Build the model.
log_mod = linear_model.LogisticRegression();
log_mod.fit(x_train_data, y_train_data);

# Make predictions - both class labels and predicted probabilities.
preds = log_mod.predict(x_test_data);
pred_probs = log_mod.predict_proba(x_test_data);
prob_pos = pred_probs.transpose()[1];  # P(X = 1) is column 1
prob_neg = pred_probs.transpose()[0];  # P(X = 0) is column 0

# Look at results.
pred_df = pd.DataFrame({'Actual':y_test_data, 'Predicted Class':preds, 'P(1)':prob_pos, 'P(0)':prob_neg});
print(pred_df.head(15));
print('Accuracy: ' + str(accuracy_score(y_test_data, preds)));
print('Precison: ' + str(precision_score(y_test_data, preds)));
print('Recall: ' + str(recall_score(y_test_data, preds)));
print('F1: ' + str(f1_score(y_test_data, preds)));
print('ROC AUC: ' + str(roc_auc_score(y_test_data, preds)));
print("Confusion Matrix:\n" + str(confusion_matrix(y_test_data, preds)));

# --------------------------------------------------------- #
# --- Section 7: Random Forest Evaluation ----------------- #
# --------------------------------------------------------- #

# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [5, 10, 50, 100];
depth = [3, 6, None];
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp);
		mod.fit(x_train_data, y_train_data);

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test_data);
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n) + ', depth =' + str(dp) + ' -------------------');
		# Look at results.
		print_multiclass_classif_error_report(y_test_data, preds);

# ------------------------------------------------------------ #
# --- Section 8: Random Forest Against the Validation Data --- #
# ------------------------------------------------------------ #

n_est = [5, 10, 50, 100];
depth = [3, 6, None];
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp);
		mod.fit(x_train_data, y_train_data);

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test_vald);
		print('---------- COMPARING RF MODEL: n_estimators = ' + str(n) + ', depth =' + str(dp) + ' -------------------');
		# Look at results.
		print_multiclass_classif_error_report(y_test_vald, preds);