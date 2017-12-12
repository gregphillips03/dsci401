import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import itertools;
from sklearn.cross_validation import KFold;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import precision_score;
from sklearn.metrics import recall_score;
from sklearn.metrics import f1_score;
from sklearn.metrics import roc_auc_score;
from sklearn.metrics import classification_report;
from sklearn.metrics import confusion_matrix;
import warnings;
warnings.filterwarnings('ignore'); 

# Get a list of the categorical features for a given dataframe. M
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe))

# Get the indices of the categorical features for a given dataframe. 
def cat_feature_inds(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	enums = zip(list(dataframe), range(len(list(dataframe))))
	selected = filter(lambda (name, ind): not(dataframe[name].dtype in [td['a'].dtype, td['b'].dtype]), enums)
	return map(lambda (x, y): y, selected)

# Print out common error metrics for the binary classifications.
def print_binary_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Precison: ' + str(precision_score(y_test, preds)))
	print('Recall: ' + str(recall_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))
	
# Print out common error metrics for the binary classifications.
def print_multiclass_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Avg. F1 (Micro): ' + str(f1_score(y_test, preds, average='micro')))
	print('Avg. F1 (Macro): ' + str(f1_score(y_test, preds, average='macro')))
	print('Avg. F1 (Weighted): ' + str(f1_score(y_test, preds, average='weighted')))
	print(classification_report(y_test, preds))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))

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

#Runs cross validation models, predicts classes
def run_cv(X, y, clf_class, **kwargs):
	kf = KFold(len(y), n_folds=5, shuffle=True);
	y_pred = y.copy();

	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]; 
		y_train = y[train_index]; 
		clf = clf_class(**kwargs); 
		clf.fit(X_train, y_train); 
		y_pred[test_index] = clf.predict(X_test); 

	return y_pred;

#Simple accuracy return to be nested with run_cv
def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred) 

#pretty way to draw a confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

#Runs cross validation models, and predicts probabilities 
def run_prob_cv(X, y, clf_class, **kwargs):
	kf = KFold(len(y), n_folds=5, shuffle=True)
	y_prob = np.zeros((len(y), 2))
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train = y[train_index]
		clf = clf_class(**kwargs)
		clf.fit(X_train, y_train)
		y_prob[test_index] = clf.predict_proba(X_test)
	return y_prob