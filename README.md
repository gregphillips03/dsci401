<h2><b>Assignment 3</b></h2>
<br>
Author: Greg Phillips</br>   
Date:   01.12.2017
<hr>

Assignment 3: Customer Churn
==============================================


1) Load the “churn_data.csv” dataset into Python. What is the response variable, and what are the predictor variables?
-----------------------------------------------------------------------------------------------------------

+ Response Variable: 
	
	+ 'Churn' or the Yes/No boolean value of whether or not a customer left for a competitor. 

+ Predictor Variables:

	+ Gender
	+ Age
	+ Income
	+ FamilySize
	+ Education
	+ Calls
	+ Visits


2) What data transforms are necessary to perform on this data and why?
----------------------------------------------------------------------

+ It's important to change the response variable from text to something numeric. I chose to change it outside of the work environment. 

	+ If the value was 'Yes', I changed it to a 1. 

	+ If the value was 'No', I changed it to a 0.

+ The data is also comprised of numeric and text values. In order to use it properly, it was one-hot encoded using the pd.getdummies method. 


3) What modeling approaches did you use and why? Describe your model development process, including the different models tried, feature selection methods, and the different transformation techniques you employed. 
--------------------------------------------------------------------------------------------------------

+ I first started off with K-nearest

	+ I thought it might be useful to see if it was a "near is like" problem. 

	+ K-nearest results were actually really ugly. 

4) Which error metrics did you use to assess performance and why? What kind of performance did you obtain
on the different models you built?
--------------------------------------------------------------------------------------------------------

+ I used:

	+ Accuracy

	+ Precision

	+ Recall

	+ F1

	+ Area Under Curve (AUC)

+ I mostly focused on the AUC, to show how much the model actually got right. 

+ The K-nearest approach didn't work very well. The amount underneath the curve was only a bit better than half, which isn't very good

+ Even the confusion matrix is poor, as shown below:


```python
'''
Accuracy: 0.576923076923
Precison: 0.666666666667
Recall: 0.307692307692
F1: 0.421052631579
ROC AUC: 0.576923076923
Confusion Matrix:
[[11  2]
 [ 9  4]]
'''
 ```

