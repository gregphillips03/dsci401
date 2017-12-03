<h2><b>Assignment 3</b></h2>
<br>
Author: Greg Phillips</br>   
Date:   01.12.2017
<hr>

Assignment 3: Customer Churn
==============================================

<hr>

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

<hr>

2) What data transforms are necessary to perform on this data and why?
----------------------------------------------------------------------

+ It's important to change the response variable from text to something numeric. I chose to change it outside of the work environment. 

	+ If the value was 'Yes', I changed it to a 1. 

	+ If the value was 'No', I changed it to a 0.

+ The data is also comprised of numeric and text values. In order to use it properly, it was one-hot encoded using the pd.getdummies method. 

<hr>

3) What modeling approaches did you use and why? Describe your model development process, including the different models tried, feature selection methods, and the different transformation techniques you employed. 
--------------------------------------------------------------------------------------------------------

+ I first started off with K-nearest

	+ I thought it might be useful to see if it was a "near is like" problem. 

	+ K-nearest results were actually really ugly.

<hr> 

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


+ I switched to a 2-Class Logistic Regression approach. It gave a negligibly better improvement, but still fell very short of anything useful. 

+ It encompasses more underneath the curve, but it's still not even good enough for government work. Confusion matrix also improves. 

```python
'''
Accuracy: 0.653846153846
Precison: 0.7
Recall: 0.538461538462
F1: 0.608695652174
ROC AUC: 0.653846153846
Confusion Matrix:
[[10  3]
 [ 6  7]]
'''
```

+ Then I stepped away from the code a bit an thought about the overall question. 

	+ An important question to ask should be, when a customer churns, how often does my classifier predict that correctly? In other words, we're really concerned about <b>recall</b>. 

	+ Also, <b>precision</b> or, <i>When a classifier predicts a customer will churn, how often does that individual actually churn?</i> The differences between those two questions is small, but it really make a huge difference if you sit an think it through. 

<hr>

5) Construct the best (i.e. least-error) possible model on this data set. What are the predictors used?
-------------------------------------------------------------------------------------------------------

<hr>

6) Load the dataset “churn_validation.csv” into a new data frame and recode as necessary. Predict the
outcomes for each of the customers and compare to the actual. What are the error rates you get based on
your selected metrics?
-------------------------------------------------------------------------------------------------------

<hr>

7) Consider the best model you built for this problem. Is it a good model that can reliably be used for
prediction? Why or why not?
-------------------------------------------------------------------------------------------------------