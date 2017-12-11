<h2><b>Assignment 4</b></h2>
<br>
Author: Greg Phillips</br>   
Date:   11.12.2017
<hr>

Assignment 4: Predicting Employees and Contractors
==============================================

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

<h2><b>2) Cleaning up the Data</b></h2>

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
data = pd.read_excel('./data/incs.xlsx', sheet_name='ALL');
```

+ The ```read_excel``` method takes a string argument for the worksheet you want to import. There are other paramaters you can specify as well. Since my spreadsheet contained multiple tabs within it, I used the ```sheet_name='ALL'``` argument as well. 

