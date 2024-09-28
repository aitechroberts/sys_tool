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