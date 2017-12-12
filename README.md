<h2><b>Assignment 4</b></h2>
<br>
Author: Greg Phillips</br>   
Date:   11.12.2017
<hr>

Assignment 4: Predicting Employees and Contractors
==============================================

<img src="./img/sfl.PNG" title="Safety For Life Logo" alt="sfl logo" style="display: block; margin: auto;" />

<hr>

<h2><b>0) The Question</b></h2>

<hr>

+ TODO Question information. 

<hr>

<h2><b>1) The Data</b></h2>

<hr>

+ The data is comprised of Safety, Health, and Environmental records generated from AECOM's IndustrySafe online incident reporting system. Each time an employee experiences an incident (such as those associated with physical harm, property damage, or hazardous material spills), they - or someone on their behalf - enter the incident into the IndustrySafe database. 

+ In keeping with the Health Insurance Portability and Accountability Act of 1996 (HIPAA); best practices associated with the protection and dissemination of employee personally identifiable information (PII); and AECOM's own internal guidelines that govern the dissemination of said information, all fields which could possibily indicate a specific employee have been removed from the data set prior to importing it into the working environment. 

	+ Removal of these attributes has no bearing on the outcome of the analysis. Attributes such as employee names and employee tracking IDs would not factor into the overall outcome. 

+ The dataset contains the following attributes:

	+ Business Group
		+ Sub entity within AECOM as a whole
	+ Incident Type
		+ Classification of the incident record itself
	+ Potential Severity
		+ How bad of an outcome this could / did have
	+ Potential Probability
		+ Frequency we expect this type of incident to occur
	+ Business Line
		+ Fuctional line of business (transportation, civil infrastructure, oil & gas, etc.)
	+ Worker Type
		+ AECOM Employee or Contractor
	+ Date of Incident
		+ Calendar Day / Month / Year the incident occured

+ Overall, this data set is composed of 6510 records from fiscal year 2017. 

<hr>

<h2><b>2) Cleaning up the Data (Outside the Work Environment)</b></h2>

<hr>

+ The data required a bit of massaging and manipulation before analysis. The following transformations were made to the data outside of the work environment:

+ The data set contained the column 'Date of Incident', which was renamed to 'Month'

	+ Each record contained the specific calendar day which the incident occurred. 
	+ These fields were mapped to the corresponding month of the year, as a text label.
		+ i.e., 03/04/2017 became 'March'. 

+ Contractors can come in more than one type. That distinction is not necessary for this analysis. 

```python
'''
- Changed 'Worker Type' WHERE value == 'Contractor/Subcontractor (Managed)'
- Replaced 1040 locations with 'Contractor'

- Changed 'Worker Type' WHERE value == 'Contractor/Subcontractor (Unmanaged)'
- Replaced 1316 locations with 'Contractor'
'''
```

+ Some records do not need to be included, and were removed. The data set contains more than 'AECOM Employees' and 'Contractors'. It also contains joint venture partners, as well as members of the public. 

```python
'''
- Removed records WHERE value of 'Worker Type' == 'JV Partner'. 
- Removed a total of 6 records. 

- Removed records WHERE value of 'Worker Type' == 'Third Party/Public'. 
- Removed a total of 38 records. 
'''
```

<hr>

<h2><b>3) Importation and First Look</b></h2>

<hr>

+ It seems the business world runs on MS Excel. As such, vendors and third party software suppliers normally add in the ability to get data out of an application in a .xlsx format. It's convenient for normal end users, but it isn't very conducive to analysis.

+ You can use Excel's built-in save function to save the file as a .csv extension; however, this doesn't always work. Depending on the exported format (JSON, XML, HTML), the data might come out a bit messy. 

+ Fortunately, pandas and Python have a built in module for importing directly from a spreadsheet.

+ From a terminal (substitute pip for the package manager you use): 

```dos
pip install xlrd
```

+ Then within your Python code:

```python
data = pd.read_excel('./data/incs.xlsx', sheet_name='fy17');
```

+ This comes in especially useful if your spreadsheet isn't hosted on you local machine. You can specify a host within the string to point pandas to a remotely hosted spreadsheet. YMMV.

+ The ```read_excel``` method takes a string argument for the worksheet you want to import. There are other paramaters you can specify as well. Since my spreadsheet contained multiple tabs within it, I used the ```sheet_name='fy17'``` argument as well.

	+ Due to the size of the data, I was forced to download each month separately. The 'fy17' tab is where I aggregated each month by hand. Ergo, that's where I want pandas to look for the entirey of my incident records. 

+ Here, you can see column names within the data set. I cleverly placed 'Worker Type' at index 0, as it will function as the dependent /response variable for this analysis. 

```python
'''
Column names: 
[u'Worker Type', u'Incident Type', u'Month', u'Potential Severity', u'Potential Probability', 
u'Business Group', u'Business line']

'''
```

+ Here, a simple ```.head()``` and ```.tail()``` allow us to verify we've got the correct data in the data set. 

```python

print("\nSample data: "); 
print(data.head(5)); 

'''
Sample data: 
      Worker Type
0  AECOM Employee 
1  AECOM Employee 
2  AECOM Employee
3  AECOM Employee
4  AECOM Employee

'''
print(data.tail(5));

'''
     Worker Type
6505  Contractor
6506  Contractor
6507  Contractor
6508  Contractor
6509  Contractor  

'''
```

<hr>

<h2><b>4) Transformation and Cleaning up Data</b></h2>

<hr>

+ Now that we've verified we're able to actually read in from our spreadsheet (honestly, how cool is that?), we can move on to transforming the data into something Python can actually work with. In the real world, most data is dirty. In other words, it really requires a good bit of 'cleaning' up in the general sense before we can apply it to a model. Some stuff is apparent, while some stuff doesn't jump out at you. 

+ First, I isolated the response variable and encoded it using numpy. At this point, we have two values in the data set:

	+ 'AECOM Employee'
	+ 'Contractor'

+ Every time the code 'sees' the text 'AECOM Employee', I want it encoded with a 1 [true]. Otherwise, I want it to be encoded with a 0 [false]. 

+ First, pull out the response variable from the data frame (in this case, 'Worker Type'):

```python
target_result = data['Worker Type']; 

```

+ Then, using numpy, apply a simple if / then statement (via the ```numpy.where``` method) for the transformation:

```python
y = np.where(target_result == 'AECOM Employee', 1, 0)

```

+ This yields a response variable ('Worker Type'), that is either 'true' [1] for an AECOM Employee, or is 'false' [0] for a Contractor. 

+ Next, we need to get rid of the text values. This part might not be apparent at first, but its crucial to performing the analysis. Ergo, we need a label encoder, or something that can take a column, find all the different text values / categories in it, then encoded it properly. 

+ To accomplish this, we'll use pandas ```get_dummies()``` method, coupled with a utility label encoder borrowed from <a href=https://github.com/chrisgarcia001>Chris Garcia, PhD</a>. The ```get_dummies()``` method converts categorical values into "dummy", or indicator variables, while the ```cat_features()``` function returns a list of the categorical features for a given dataframe. They work hand in hand to get rid of the text and leave us with a nice, clean, numerical data frame. 

```python
data = pd.get_dummies(data, columns=util.cat_features(data));

```

+ For good measure, I pull out the feature space for future use. 

```python
features = feature_space.columns;  

```

+ Then transform the feature space into a sort of 'matrix of floats'. 
	
	+ Note: this probably isn't necessary for this data set, but it's something I'm accumstomed to doing. I don't believe it has a negative impact if I do it regardless, but I'm not sure if it will have a positive influence either. Since I sort of want this to be a 'plug and play' set of code, I'm leaving it in.

```python
X = feature_space.as_matrix().astype(np.float); 

```

+ What is important for some models:

	+ A lot of predictors care about the size of the features themselves, even if the actual scaling of those features is meaningless in their relationship. For instance, if I had a data set that included the number of incidents submitted by a particular person and also included the number of "really really bad" incidents they submitted (i.e., something terrible happened to them), the former would outweigh the latter by a few orders of magnitude (at least we really hope so in our line of work). However, that doesn't mean that the latter is any less significant. 

	+ To account for this, we normalize the data for each member of a feature from a range between 1.0 and -1.0. This keeps models from applying too much weight where they shouldn't. Conveniently ```sklearn.preprocessing``` includes a ```StandardScaler``` which accomplishes this for us. 

+ Apply the scaler to the X data, or the predictors:

```python
scaler = StandardScaler(); 
X = scaler.fit_transform(X);

```

+ At this point, we have a features space (which I've denoted as X), and a target value (which I have denoted as y). This is our model, and it's ready to be applied to a multitude of algorithms for predictions. 

+ However, it's good measure at this point to check and make sure we've not done anything crazy with the data. We done a lot of moving and transforming, so it's good to check and make sure that we didn't forget anything. 

+ Let's check to see if there's missing data. The ```check_missing_data``` is a utility function I built to look for missing values within a data frame. It looks for ```null``` values across the set, and prints out the column names that contain them (if it finds them). 

+ Here, we are simply printing back to the screen the name of the frame, and whether or not there was missing data. 

```python
b = util.check_missing_data(data); 
if(b):
	print('Found Missing Data'); 
	show_name(data); 
	print('\n');
else:
	print('No Missing Data!');
	show_name(data); 
	print('\n');

'''
No Missing Data!
DataFrame Name is: data
'''

```

+ The ```check_missing_data``` didn't find anything missing throughout the data set, which is a good thing. I find this utility to be really helpful, especially when the data set is large and non-conducive to 'looking it over' with your eyes. 

+ One final sanity check, before moving onto predictions. 

```python
print("Feature space contains %d records and %d columns" % X.shape); 
print("Number of Response Types:", np.unique(y)); 

```

+ Which yields:

```python
'''
Feature space contains 6510 records and 66 columns
('Number of Response Types:', array([0, 1]))
'''

```

+ Here we have two important pieces of information:

	+ Our feature space contains the correct amount of records, and after we label encoded all the text, it has the correct amount of columns as well. 

		+ Worker Type x 1
		+ Incident Type x 9
		+ Month x 12
		+ Potential Severity x 5
		+ Potential Probability x 5
		+ Business Group x 8
		+ Business Line x 26

	+ And our response variable, has the correct types we encoded earlier (either a 1 or a 0)

+ Now that we've cleaned up the data, transformed it, and given a few sanity checks, we can rest easy knowing that our data is ready to be processed. Now we can move onto making the predictions. 

<hr>

<h2><b>5) Evaluate different modeling approaches</b></h2>

<hr>