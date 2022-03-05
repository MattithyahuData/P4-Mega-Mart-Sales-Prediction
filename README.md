# 🏪 Mega Mart Sales Prediction: Project Overview
* End to end project predicting the sales of a supermarket.
* Used XGBoost regressor model for predictions 

## Table of Contents 
*   [Resources](#resources)
*   [Data Collection](#DataCollection)
*   [Data Pre-processing](#DataPre-processing)
*   [Data Warehousing](#DataWarehousing)
*   [Exploratory data analysis](#EDA)
*   [Feature Engineering](#FeatEng)
*   [ML/DL Model Building](#ModelBuild)
*   [Model Evaluation](#ModelEval)
*   [Project Management (Agile | Scrum)](#Prjmanage)
*   [Project Evaluation](#PrjEval)
*   [Looking Ahead](#Lookahead)
*   [Questions & Contact me](#Lookahead)

<a name="resources"></a>  

## Resources Used
**Python 3, PostgreSQL** 

[**Anaconda Packages:**](requirements.txt) **pandas numpy pandas_profiling ipywidgets sklearn xgboost matplotlib seaborn sqlalchemy kaggle psycopg2 ipykernel** 
Powershell command for installing anaconda packages used for this project  
```powershell
pip install pandas numpy pandas_profiling ipywidgets sklearn xgboost matplotlib seaborn sqlalchemy kaggle psycopg2 ipykernel
```

<a name="DataCollection"></a>  

## [Data Collection](Code/P4_Code.ipynb)
Powershell command for data import using kaggle API <br>
```powershell
!kaggle datasets download -d mrmorj/big-mart-sales -p ..\Data --unzip 
```
[Data source link](https://www.kaggle.com/mrmorj/big-mart-sales)
[Data](Data/train_v9rqX0R.csv)
*  Rows: 8523 / Columns: 12
    *   Item_Identifier              
    *   Item_Weight                  
    *   Item_Fat_Content              
    *   Item_Visibility             
    *   Item_Type                     
    *   Item_MRP                    
    *   Outlet_Identifier            
    *   Outlet_Establishment_Year     
    *   Outlet_Size                  
    *   Outlet_Location_Type         
    *   Outlet_Type                   
    *   Item_Outlet_Sales           
                     

<a name="DataPre-processing"></a>  

## [Data Pre-processing](Code/P4_Code.ipynb)
After I had all the data I needed, I needed to check it was ready for exploration and later modelling. I made the following changes:   
*   General NULL and data validity checks  
*   NULL values present in Item_Weight and Outlet_Size. 


<br>

```python
# Viewing the data types of the columns
data.dtypes

# Viewing dataset shape
data.shape

# 1st check for null values and datatype check 
data.info()
```
<br>

*   Mean and Mode imputation used to fill NULL values respectively.
<br>

```python
# Handling the missing values # Imputation to handle null values
# Viewing mean value of item_weight
data.Item_Weight.mean()

# Filling the missing values in item_weight with mean value
data['Item_Weight'].fillna(data.Item_Weight.mean(), inplace=True)


# Handling the missing values # Imputation to handle null values

# Viewing mean value of item_weight
data.Outlet_Size.mode()

# Getting mode of values that are not null corresponding to those values that are NULL 
# Here it shows most commonly Grocery stores and Supermarket Type1's are small and Supermarket Type2 and Supermarket Type3 are Medium so the NULL values will be filled accordingly  
mode_outletsize = data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

# Viewing mode correspondents 
mode_outletsize


# Getting True/False for NULL values in Outlet_Size for lambda function
miss_values = data['Outlet_Size'].isnull()   

# Filling the missing values in outlet_size with mode -- # Where condition is True apply lambda function for filling NULL value with the mode! 
data.loc[miss_values, 'Outlet_Size'] = data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_outletsize[x])
```

<a name="DataWarehousing"></a>

## [Data Warehousing](Code/P4_Code.ipynb)
I warehouse all data in a Postgre database for later use and reference.

*   ETL in python to PostgreSQL Database.
*   Formatted column headers to SQL compatibility.  


```python 
# Function to warehouse data in a Postgre database 
def store_data(data,tablename):
    """
    :param data: variable, enter name of dataset you'd like to warehouse
    :param tablename: str, enter name of table for data 
    """

    # SQL table header format
    tablename = tablename.lower()
    tablename = tablename.replace(' ','_')

    # Saving cleaned data as csv
    data.to_csv(f'../Data/{tablename}_clean.csv', index=False)

    # Engine to access postgre
    engine = create_engine('postgresql+psycopg2://postgres:password@localhost:5432/projectsdb')

    # Loads dataframe into PostgreSQL and replaces table if it exists
    data.to_sql(f'{tablename}', engine, if_exists='replace',index=False)

    # Confirmation of ETL 
    return("ETL successful, {num} rows loaded into table: {tb}.".format(num=len(data.iloc[:,0]), tb=tablename))
 
# Calling store_data function to warehouse cleaned data
store_data(data,"P2 Bank Churn Prediction")
```

<a name="EDA"></a>  

## [Exploratory data analysis](Code/P4_Code.ipynb) 
I looked at the distributions of the data and the value counts for the various categorical variables that would be fed into the model. Below are a few highlights from the analysis.

*   This for loop, loops over all non numeric fields to print the values where there are less than 6 unique values
<br>

```python
# Getting non numeric columns 
nonnumeric_cols = data.select_dtypes(exclude=["float", 'int']).columns.tolist()


# Checking distribution of categorical fields. For loop to iterate and print value_counts for categorical values 
for i, item in enumerate(nonnumeric_cols):
    # if there is less than 6 values then continue 
    if len(data[item].value_counts()) < 6:

        # print column name and value_counts()
        print()
        print(item)
        print(data[item].value_counts())
    # Else do nothing
    else:
        None
```

<img src="images/categoricalfeatures_countdistrib.png" />
<img src="images/categoricalfeatures_distrib.png" />

*   The features are not correlated generally.
<img src="images/correlation.png" />


<a name="FeatEng"></a>  

## [Feature Engineering](Code/P4_Code.ipynb) 
Here I fixed multiple instance issues for instances that should be the same instance.
<br>

```python
# Replacing data with duplicate instance names
data.replace({'item_fat_content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
```
I used label encoder to encode the categorical variable(s) into numeric values for compatibility with the ML model. I also split the data into train and tests sets with a test size of 20%.
*   Label encoder encoding to encode the categorical non-numeric values
<br>

```python
# Checking distribution of categorical fields. For loop to iterate and print value_counts for categorical values 
for i, item in enumerate(nonnumeric_cols):

    # Encoding categorial columns 
    data[item] = le.fit_transform(data[item])


# Splitting dependent and independent features to apply scaling
X = data.drop(['item_outlet_sales'], axis=1)

# Dependent feature | Target variable 
y = data['item_outlet_sales']


# Using train test split to split train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=23, shuffle=True)

# Viewing shape of train / test data
print(X_train.shape)
print(X_test.shape)
```


<a name="ModelBuild"></a> 

## [ML/DL Model Building](Code/P4_Code.ipynb)

I applied the XGBRegressor model to achieve the predictions. 

```python
# Calling XGBRegressor for the regression use case 
regressor = XGBRegressor()

# Training model on training data byy fitting it with train data
regressor.fit(X_train, y_train)
```


<a name="ModelEval"></a> 

## [Model Evaluation](Code/P4_Code.ipynb)
*   I used the r2_score to see the error associated with the model. But because it is a regression use case I can’t give an accuracy score. 
An R-Squared value above 0.7 would generally be seen as showing a high level of correlation. The model achieved a R2 value of 0.546.
A value of 0.5 means that half of the variance in the outcome variable is explained by the model.

*   Plotting the actual and predicted values for botht the training and test sets shows how accracy and linear correlation decreases in the test data. 
<img src="images/trainevaluation.png" />
<img src="images/testevaluation.png" />


<a name="Prjmanage"></a> 

## [Project Management (Agile/Scrum/Kanban)](https://www.atlassian.com/software/jira)
* Resources used
    * Jira
    * Confluence
    * Trello 

<a name="PrjEval"></a> 

## [Project Evaluation]() 
*   WWW
    *   The end-to-end process
    *   The review and process of a regression use case 
*   EBI 
    *   Better project management and planning would have made this project faster
    *   Explore use of other models 

<a name="Lookahead"></a> 

## Looking Ahead
*   How can I predict quarter and year on year returns? 
*   Help model accuracy by using better data 

<a name="Questions"></a> 

## Questions & Contact me 
For questions, feedback, and contribution requests contact me
* ### [Click here to email me](mailto:contactmattithyahu@gmail.com) 
* ### [See more projects here](https://mattithyahudata.github.io/)

 
