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

+ I first started off with <b>K-nearest</b>

	+ I thought it might be useful to see if it was a "near is like" problem. 

	+ K-nearest results were actually really ugly.

+ Then I tried the <b>2-Class Logistic Regression</b>

	+ Since we have two classes, I gave this one a try. 

	+ The results here were only marginally better than K-nearest. 

+ Finally, I went with the <b>Random Forest</b> just as I thought the problem itself might be more related to a if/then problem than the 2 former approaches. 
	
	+ The approach proved to be the best. 

+ A last point of note, and something I'll use on the final project, relates to how predictors view features. A lot of predictors actually care about the relative size of features, even though those scales make little or no sense. Here, I used a standard scaler across the feature space just to ensure the model doesn't do anything crazy underneath the hood. 

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

+ The <b>K-nearest</b> approach didn't work very well. The amount underneath the curve was only a bit better than half, which isn't very good

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


+ I switched to a <b>2-Class Logistic Regression</b> approach. It gave a negligibly better improvement, but still fell very short of anything useful. 

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

+ <b><i>Then I stepped away from the code a bit</i></b> and thought about the overall question. 

	+ An important question to ask should be, <i>'when a customer churns, how often does my classifier predict that correctly?'</i> In other words, we're really concerned about <b>recall</b>. 

	+ Also, <b>precision</b> or, <i>'when a classifier predicts a customer will churn, how often does that individual actually churn?'</i> The differences between those two questions is small, but it really make a huge difference if you sit an think it through. 

+ After thinking this over, and determining that these were really the error metrics to concentrate on, I moved to a random forest evaluation. 

<hr>

5) Construct the best (i.e. least-error) possible model on this data set. What are the predictors used?
-------------------------------------------------------------------------------------------------------

+ The <b>Random Forest</b> model blows <b>K-nearest</b> and <b>2-Class Logistical Regression</b> out of the water. 

```python
'''
---------- EVALUATING MODEL: n_estimators = 5, depth =3 -------------------
Accuracy: 0.961538461538
Avg. F1 (Micro): 0.961538461538
Avg. F1 (Macro): 0.961481481481
Avg. F1 (Weighted): 0.961481481481
             precision    recall  f1-score   support

          0       1.00      0.92      0.96        13
          1       0.93      1.00      0.96        13

avg / total       0.96      0.96      0.96        26

Confusion Matrix:
[[12  1]
 [ 0 13]]
'''

```

+ We get really good <b>precision</b> and <b>recall</b> scores, supported visually by the high F1 score as well. Even the confusion matrix is top-notch. It doesn't get everything right; however, we are still only using a very small data set. 

<hr>

6) Load the dataset “churn_validation.csv” into a new data frame and recode as necessary. Predict the
outcomes for each of the customers and compare to the actual. What are the error rates you get based on
your selected metrics?
-------------------------------------------------------------------------------------------------------

+ Performance varies. I ran it against the same inputs that we use to build the model, and it really depends on which way the data is split up. 

```python
'''
---------- EVALUATING MODEL: n_estimators = 5, depth =6 -------------------
Accuracy: 0.857142857143
Avg. F1 (Micro): 0.857142857143
Avg. F1 (Macro): 0.787878787879
Avg. F1 (Weighted): 0.839826839827
             precision    recall  f1-score   support

          0       0.83      1.00      0.91         5
          1       1.00      0.50      0.67         2

avg / total       0.88      0.86      0.84         7

Confusion Matrix:
[[5 0]
 [1 1]]
'''
```

<hr>

7) Consider the best model you built for this problem. Is it a good model that can reliably be used for
prediction? Why or why not?
-------------------------------------------------------------------------------------------------------

+ I think the model itself is solid. We see that in when we build it. We got really good metrics in some cases, and fair metrics in others. All in all, we know that the predictors we've chosen actually relate to whether or not a customer leaves. Recall and Precision (and thus F1), really speak to this. 

+ However, this is an ML problem; ergo, it takes time to learn. The initial model was only trained with ~85 records, which honestly isn't enough to really allow the model to accurately learn. Once it runs into unseen data, it gets a little bit sketchy on performance. 

+ To directly answer the question, yes it is a good model to use for prediction, but we really need more records to properly train the model, prior to deployment.  