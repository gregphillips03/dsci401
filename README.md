Assignment 2: Regression Modeling & Prediction
==============================================

I. Data Preparation Questions
-----------------------------

1) What specific data transforms did you perform prior to exploration and analysis, and why did you choose these?
-------------------------------------------------------------------------------------------

+ I first got rid of the PID attribute. It's numeric, but it has no bearing on the sales price. Instead, it's simply a unique identifer of key used to differentiate records. 

+ Some things irk me, and some things are simply carried over by best practice. I removed the sales price as the last column (because it irks me to not be on the far left in a tabular environment). I also replaced all the '.' characters in the column headers with underscores, since some environments get confused when parsing strings that contain '.' in them. 

+ I performed a large amount of data cleaning. Naturally that's a huge part of the overall process. In this case, it mostly involved filling in fields where the data had been encoded as 'NA' during capture. This seemed to cause an issue with either Python or sklearn, where the machine viewed those values as invalid. 

+ The dataset was also comprised of both numeric and categorical data. In order to use all the data, the categorical values were given binary identifiers.

+ Here is the actual data cleanup that occurred outside of the working environment:

```python
/- Begin Cleaning Up Test Data -/

- imputed Lot.Frontage WHERE value == NA
- I used the average (69.2) to fill in the missing values

- House without Alley -

- imputed Alley WHERE value == NA
- Python thinks NA is a missing value, but in the data, this is being used categorically as
- a distinction 
- replaced NAs with No, meaning no Alley

- House without Veneer -

- imputed Mas.Vnr.Type WHERE value == blank
- replaced 17 blank locations with None, since None was part of the data set

- imputed Mas.Vnr.Area WHERE value == NA
- Python thinks NA is a missing value, but in the data, this is being used categorically as
- a distinction
- same for imputations below
- replaced 17 NA with 0, since 0 was part of the data set 

- House without Basement - 

- imputed Bsmt.Qual WHERE value == blank
- replaced 1 location with TA as a quick pivot table revealed that to be the most frequent value
- without an ontolgoy or data dictionary its hard to tell what to use, ergo most frequent

- imputed Bsmt.Qual WHERE value == NA
- replaced 60 locations with None

- imputed Bsmt.Cond WHERE value == blank
- replaced 1 location with TA as a quick pivot table revealed that to be the most frequent value
- without an ontolgoy or data dictionary its hard to tell what to use, ergo most frequent

- imputed Bsmt.Cond WHERE value == NA
- replaced 60 locations with None

- imputed Bsmt.Exposure WHERE value == blank
- replaced 4 locations with No

- imputed Bsmt.Exposure WHERE value == NA
- replaced 60 locations with No

- imputed Bsmt.Fin.Type.1 WHERE value == NA
- replaced 60 locations with None

- imputed Bsmt.Fin.Type.1 WHERE value == blank
- replaced 1 location with None

- imputed Bsmt.Fin.SF.1 WHERE value == NA
- replaced 1 location with 0

- imputed Bsmt.Fin.Type.2 WHERE value == NA
- replaced 60 locations with None

- imputed Bsmt.Fin.Type.2 WHERE value == blank
- replaced 2 locations with None

- imputed Bsmt.Fin.SF.2 WHERE value == NA
- replaced 1 location with 0

- imputed Bsmt.Unf.SF WHERE value == NA
- replaced 1 location with 0

- imputed Total.Bsmt.SF WHERE value == NA
- replaced 1 location with 0

- imputed Bsmt.Full.Bath WHERE value == NA
- replaced 2 locations with 0

- imputed Bsmt.Half.Bath WHERE value == NA
- replaced 2 locations with 0

- House without Fireplace -

- imputed Fireplace.Qu WHERE value == NA
- replaced 1134 locations with No

- House without Garage -

- inputed Garage.Type WHERE value == NA
- replaced 119 locations with None

- imputed Garage.Yr.Blt WHERE value == NA
- this one was a bit tricky, but I ended up deciding to use the avg (year 1977)
- replaced 121 locations with 1977

- imputed Garage.Yr.Blt WHERE value == 2207
- considered this as a data entry error
- replaced 1 location with 2007

- imputed Garage.Finish WHERE value == NA
- replaced 119 locations with None

- imputed Garage.Finish WHERE value == blank
- replaced 2 locations with None

- imputed Garage.Cars WHERE value == NA
- replaced 1 location with 0

- imputed Garage.Area WHERE value == NA
- replaced 1 location with 0

- imputed Garage.Qual WHERE value == blank
- replaced 1 locatoin with TA as a quick pivot table revealed that to be the most frequent value
- without an ontolgoy or data dictionary its hard to tell what to use, ergo most frequent

- imputed Garage.Qual WHERE value == NA
- replaced 120 locations with None

- imputed Garage.Cond WHERE value == blank
- replaced 1 location with TA as a quick pivot table revealed that to be the most frequent value
- without an ontolgoy or data dictionary its hard to tell what to use, ergo most frequent

- imputed Garage.Cond WHERE value == NA
- replaced 120 locations with None

- House without Pool -

- imputed Pool.QC WHERE value == NA
- replaced 2335 locations with None

- House without Fence - 

- imputed Fence WHERE value == NA
- replaced 1880 locations with None

- House without Misc Qualities - 

- imputed Misc.Feature WHERE value == NA
- replaced 2260 locations with None 

/- End Cleaning Up Test Data -/

/- Begin Cleaning Up Validation Data -/

- imputed Lot.Frontage WHERE value == NA
- I used the average (69.2) to fill in the missing values

- House without Alley -

- imputed Alley WHERE value was listed == NA
- Python thinks NA is a missing value, but in the data, this is being used categorically as
- a distinction 
- replaced 554 NAs with No, meaning no Alley

- House without Veneer -

- imputed Mas.Vnr.Type WHERE value == blank
- replaced 6 blank locations with None, since None was part of the data set

- imputed Mas.Vnr.Area WHERE value == NA
- Python thinks NA is a missing value, but in the data, this is being used categorically as
- a distinction
- same for imputations below
- replaced 6 NA with 0, since 0 was part of the data set 

- House without Basement - 

- imputed Bsmt.Qual WHERE value == NA
- replaced 19 locations with None

- imputed Bsmt.Cond WHERE value == NA
- replaced 19 locations with None

- imputed Bsmt.Exposure WHERE value == NA
- replaced 19 locations with No

- imputed Bsmt.Fin.Type.1 WHERE value == NA
- replaced 19 locations with None

- imputed Bsmt.Fin.Type.2 WHERE value == NA
- replaced 19 locations with None

- House without Fireplace -

- imputed Fireplace.Qu WHERE value == NA
- replaced 288 locations with No

- House without Garage -

- inputed Garage.Type WHERE value == NA
- replaced 38 locations with None

- imputed Garage.Yr.Blt WHERE value == NA
- this one was a bit tricky, but I ended up deciding to use the avg (year 1979)
- replaced 38 locations with 1979

- imputed Garage.Finish WHERE value == NA
- replaced 38 locations with None

- imputed Garage.Qual WHERE value == NA
- replaced 38 locations with None

- imputed Garage.Cond WHERE value == NA
- replaced 38 locations with None

- House without Pool -

- imputed Pool.QC WHERE value == NA
- replaced 582 locations with None

- House without Fence - 

- imputed Fence WHERE value == NA
- replaced 478 locations with None

- House without Misc Qualities - 

- imputed Misc.Feature WHERE value == NA
- replaced 564 locations with None 

- Electrical - 

- imputed Electrical WHERE value == NA
- replaced 1 location with SBrkr, as a quick pivot showed that to be the most frequent value

```

II. Exploratory Analysis Questions
----------------------------------

Perform an exploratory analysis on your data by visualizing and/or applying other means of data exploration.

1) What (if any) insights jump out at you?
------------------------------------------

+ I ran a heat map in Seaborn. The pairs plot was a bit messy given the amount of data present in the set. Therefore, I opted to parse down the data to variables I *thought* would have a correlation or bearing on the sales price. 

<img src="./fig/heat.png" title="Correlation Heatmap" alt="Corrleation Heatmap" style="display: block; margin: auto;" />

+ Initially, the overall quality, total basement square footage, and great room living area had high correlations with the sales price - at least based off of the heatmap. 

2) Do you have any hypotheses about relationship of certain variables to the price?
-----------------------------------------------------------------------------------

+ I thougtht that some of the variables would reinforce each other, ergo multicollinearity. This didn't pan out the way I thought that it would, though. I thought that possibly the size of a room, or the condition of a feature would feed into the overall quality, but this strangely wasn't the case. 

+ I still think there is a small case to argue the multicollinearity exist, especially after running a variance inflation factor test. Some of the factors resolve to infinity, but I'm not sure if that is due to a div/0 problem 'under the hood' or not. To stay on the safe side, I left them in the set. I would need to know a bit more about how it actually works under the hood to really tell.

+ VIF Test I Used: 

```python
df2 = pd.read_csv('./data/AmesHousingSetAv2.csv');
df2 = df2.drop('PID', axis = 1); 
#drop out missing vals 
df2.dropna(); 
#drop non-numeric cols
df2 = df2._get_numeric_data();
drop_periods(df2); 
features = list(df2); 
features.remove("SalePrice"); 
str1 = " + ".join(features); 
#break into left and right hand side; y and X
#get y and X dataframes based on this regression:
#so sales price 'is dependent up on all the features in the string'
y, X = dmatrices(formula_like='SalePrice ~ ' + str1, 
	data=df2, return_type='dataframe');
vif = pd.DataFrame();
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])];
vif["features"] = X.columns;
```

+ These are the variables that resolved to infinity:

```python
BsmtFin_SF_1
BsmtFin_SF_2
Bsmt_Unf_SF
Total_Bsmt_SF
X1st_Flr_SF
X2nd_Flr_SF
Low_Qual_Fin_SF
Gr_Liv_Area
```

III. Model Building
-------------------

First construct a baseline model (containing all predictors) to predict the price. Then build the
best model you can devise. In this part use ONLY dataset A and DO NOT TOUCH dataset B.
You will want to split this into training and test sets and apply error metrics/compare models
only on the test data.

1) What approach did you use to arrive at the best model? Why did you select this approach?
-------------------------------------------------------------------------------------------

+ I first built a base model using regular regression. This actually produce a *useable* model, with a decent R^2 score.

```python
#output of 
'''
MSE = 699316040.50473368
MAE = 11746.06613336131
R^2 = 0.89683828997227877
EVS = 0.89788927124721352

'''
#not terrible but I'm sure there is a better model
```

+ Next I moved on to the K-best approach. R^2 went down a bit, but the MSE jumped considerably. 

```python
#output of 
'''
MSE = 892508249.80567992
MAE = 14630.935672674328
R^2 = 0.86833895988236054
EVS = 0.86895928720374904

'''
#R^2 went down a bit, but MSE went up a lot
```

+ I felt like lasso was the best approach from the beginning, so I settled on this model. 

```python
'''
R^2 (Lasso Model with alpha=5.0): 0.912504625364
R^2 (Lasso Model with alpha=5.1): 0.912572532422
R^2 (Lasso Model with alpha=5.2): 0.912580801171
R^2 (Lasso Model with alpha=5.3): 0.91271073246
R^2 (Lasso Model with alpha=5.4): 0.912847860123
R^2 (Lasso Model with alpha=5.5): 0.912985183824
R^2 (Lasso Model with alpha=5.6): 0.913108884678
R^2 (Lasso Model with alpha=5.7): 0.913101134751
R^2 (Lasso Model with alpha=5.8): 0.913089623924
R^2 (Lasso Model with alpha=5.9): 0.913082246264
'''

#we've reached our limit @ alpha == 5.6
#this gives us the best overall R^2 score
#i'll use this for the actual model
```


2) Which error metric(s) are you using to compare performance? What is the value(s) of the error metric(s)for the baseline model and your best model?
----------------------------------------------------------------------------------------------------------

+ Primarily, I relied on the R2 score and Explained Variance Score. Since there were such a large amount of variable, and not surefire way to decide what to throw away, I wanted to explain variation the best that I could. 

IV. Predicting and Validating
-----------------------------

Run your baseline and best models on dataset B. DO NOT do any further training. Remember to apply all transforms you used in building the model to this set (use the transform function on the preprocessors you created in part I).

1) What are the respective error metric values for each model on this set? How did your best model do on this data as compared to the baseline?
--------------------------------------------------------------------------------------------------------

+ My best model actually performed better than the baseline model, which is rather unnerving to me. Specifically, because there are certain instances that, at least in my opinion, the model performs rather bad at.

2) Is your best model a good model? Why or why not
--------------------------------------------------

+ I'll argue that it is a good model for *generalized* use. I really only think this is borderline *fair* for use in actual applications. 
