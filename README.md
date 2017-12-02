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

3) What modeling approaches did you use and why? Describe your model development process, including the different models tried, feature selection methods, and the different transformation techniques you employed. 
---