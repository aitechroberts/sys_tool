# Data Engineering for MLOps

## Data Cleaning and Preprocessing

You often must clean and order various "dirty" data for machine learning
- Ex: Cars as Automatic or Manual, or Years: 2020, 2021, 2022, etc.
    - Types could be categorical, but years and the like would be ordinal

### Data Ingestion
Here done using CSVs, but mostly will be done from a Data Lake and most likely held in cloud storage if not streaming or pulling from batches

### Data Cleaning
- Often means Columns as well so removing special characters with underscores or camel case
- Drop Columns of no interest and use the correct data types
    - Ex: use .withColumnRenamed() method to create a new column, .cast() method to proper datatype, then .drop() to drop old column followed by .distinct() to delete duplicate rows
    - .drop() is very fast because you just switch  pointer in the system
    - However, this could be done as well if no one else needs to access the data during the transformation: 
    ```python df = df.withColumn('column_name', col('column_name').cast('desired_data_type')).dropDuplicates()
    ```

- Handling missing values or Null values
    - Ex: If 50% is missing or Null in Column, then drop. Drop rows with missing values only if it does not significantly reduce your sample size

## Feature Engineering
Process:
- Classify variables into numerical and categorical
- Look at numerical summaries of numerical variables
    - Examine the distribution
- Study relationships among variables

### Processing Variables
Categorical:
- Binary (0/1, True/False), Ordinal (low/medium/high or ranking), Nominal unordered like colors
- Create summary table of descriptive statistics

Analysis:
- Create a boxplot for numerical columns
    - Must first convert to Pandas using .toPandas()
    - Generally 3-4 variables at a time
        - Ideally, choose them based on the grouping of the descriptive statistics to better visualize data and outliers
- IQR:Interquartile Range
    - IQR = Q3-Q1
    - Min = Q1 - 1.5 * IQR, Max = Q3 + 1.5 * IQR
    - Generally drop a row with 4 or more outliers
```python
[item.get_ydata()[1] for item in figure_subset['whiskers']] # or figure_subset['fliers']
```
You'll want to create a new column and drop an old column as discussed earlier as better for large-scale and distributed systems

Efficiency and Quality:
- Eliminate columns with high correlation and only pick 1
- Cast True/False to 0/1 with .cast("integer")
    - Why not cast to Boolean Type rather than Integer Type?

### String Indexer and One-Hot Encoder
Handling Nominal Values using One-Hot Encoding
- Create a vector of size = num_possible_values where 1 is the value of that category in the the vector

## Creating Models
- Create a Pipeline in Spark to handle the transformations and end on One-Hot Encoding (OHE)
- Combine features into single vector using VectorAssembler
- Then put the vectorized features into StandardScaler which just scales based on Z-score
