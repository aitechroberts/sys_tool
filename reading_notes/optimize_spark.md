# Optimizing Spark Operations
Spark is designed for distributed data processing, but optimizing its operations is crucial when working with large datasets.

## Understanding Spark's Memory and Execution Model
**Driver and Executors**: Spark applications consist of a driver program and executors on worker nodes.
Partitions: DataFrames are divided into partitions, which are the units of parallelism in Spark.
## Repartitioning DataFrames
1. Why Repartition?
**Improves Parallelism:** By increasing the number of partitions, you allow Spark to process data in parallel across multiple cores.
**Optimizes Resource Utilization**: Prevents data skew where some partitions are significantly larger than others.
2. How to Repartition
    - Use the repartition() method on your DataFrame. **Rule of Thumb**: 2-4 times the number of CPU cores.
    - **Consider Data Size**: Larger datasets may benefit from more partitions.
```python
# Assume you have 4 CPU cores, starting point 8 partitions
df = df.repartition(8)
```
3. Caveats
**Over-Partitioning**: Too many partitions can lead to overhead in task scheduling.
**Under-Partitioning**: Too few partitions can result in inefficient resource usage.

## Persisting (Caching) DataFrames
1. What Is Persisting?
**Persisting/Caching**: Stores the DataFrame in memory (or on disk) across operations, so it doesn't need to be recomputed.
- Speeds up iterative algorithms and repeated access to the same data.
2. When to Persist
- **Reusing Data**: If you perform multiple actions on the same DataFrame.
- **Iterative Algorithms**: Machine learning algorithms that iterate over the same data.
3. How to Persist
Use the cache() or persist() methods.
    - Unpersisting DataFrames: To free up memory when the DataFrame is no longer needed.

Example:
```python
# Cache in memory
df = df.cache()

# Or specify storage level
from pyspark import StorageLevel
df = df.persist(StorageLevel.MEMORY_AND_DISK)
df.unpersist()
```
4. Potential Issues
**Memory Usage**: Persisting large DataFrames can consume significant memory.
**Out of Memory Errors**: Can occur if there's not enough memory to hold the cached data.
5. Best Practices
**Selective Caching**: Only cache DataFrames that are reused multiple times.
**Adjust Storage Level:** If memory is limited, use MEMORY_AND_DISK to spill to disk when necessary.

## Understanding Spark Memory Management
1. Components of Spark Memory
**Storage Memory**: Used for caching DataFrames.
**Execution Memory**: Used for computation (e.g., shuffles, joins, aggregations).
2. Configuring Spark Memory
Adjust spark.driver.memory and spark.executor.memory:
```python
spark = SparkSession.builder \
    .appName("AppName") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
```
- Note: Adjust the memory settings based on your system's available resources.
3. Monitoring and Debugging
Spark UI: Access the Spark Web UI (usually at http://localhost:4040) to monitor jobs, stages, and memory usage.

### Example Workflow Incorporating Optimizations
Full Code Example:
```python
# Adjust Spark Session Configuration
spark = SparkSession.builder \
    .appName(APPNAME) \
    .master(MASTER) \
    .config('spark.driver.host', '127.0.0.1') \
    .config("spark.jars", JDBC_JAR_PATH) \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

df = spark.read.csv(CSV_FILE)
# Repartition to optimize writes
num_partitions = 8  # Adjust based on your system
df = df.repartition(num_partitions)

# Proceed to write to PostgreSQL
df.write.jdbc(url=JDBC_URL, table="mqtt", mode=write_mode, properties=DB_PROPERTIES)
```
## Additional Tips
1. Use Bulk Inserts with JDBC
Batch Size: Adjust the JDBC batch size to control how many rows are inserted per batch. Larger batch sizes can improve write performance but may increase memory usage. Example:
```python
df.write.jdbc(
    url=JDBC_URL,
    table="mqtt",
    mode=write_mode,
    properties=DB_PROPERTIES,
    batchsize=10000  # Adjust based on your performance tests
)
``` 
2. Monitor PostgreSQL During Writes
Use htop or top: Monitor CPU and memory usage.
Check Disk I/O: High disk I/O can become a bottleneck.
3. Indexing in PostgreSQL
Create Indexes After Data Load: Delay creating indexes until after bulk data loads to speed up inserts. Example:
```sql
CREATE INDEX idx_mqtt_column ON mqtt (column_name);
```
4. Vacuum and Analyze
After large data loads, run VACUUM ANALYZE to optimize the database.
```sql
VACUUM ANALYZE mqtt;
```
## Understanding How Spark Works with Memory and Cache
1. Spark's Lazy Evaluation
Transformation vs. Action:
Transformations: Operations that define a new DataFrame from an existing one (e.g., select, filter). They're lazily evaluated.
Actions: Operations that trigger execution (e.g., show, write).
2. Execution Plan
DAG (Directed Acyclic Graph): Spark builds a DAG of transformations.
- **Stages and Tasks**: The DAG is broken into stages, which are executed across the cluster.
3. Memory Management
- **Garbage Collection**: Spark relies on JVM garbage collection. Large objects can lead to long GC pauses.
- **Serialization**: Data is serialized during shuffles and network transfers.
4. Shuffle Operations
Operations like repartition, join, and groupBy involve shuffling data across the network.
- Optimizing Shuffles:
    - Minimize Shuffles: Design your computation to reduce the need for shuffles.
    - Use coalesce() When Reducing Partitions: Unlike repartition(), coalesce() avoids a full shuffle.
Example:

```python
df = df.coalesce(4)  # Reduce to 4 partitions without shuffle
```

## Putting It All Together
Here's how you might structure your code with these optimizations:

```python
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# Define dataset and Spark constants
APPNAME = 'Roberts_Systool_Project'
MASTER = 'local'
JDBC_JAR_PATH = "/path/to/postgresql-42.7.4.jar"
JDBC_URL = "jdbc:postgresql://127.0.0.1:5432/mqttdb"
DB_PROPERTIES = {
    "user": "postgres",
    "password": "postgres_pw",
    "driver": "org.postgresql.Driver"
}
DATA_FOLDER = "./data/FINAL_CSV"

# Initialize Spark session with optimizations
spark = SparkSession.builder \
    .appName(APPNAME) \
    .master(MASTER) \
    .config('spark.driver.host', '127.0.0.1') \
    .config("spark.jars", JDBC_JAR_PATH) \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

data_files = {
    'train70.csv': 'train',
    'test30.csv': 'test'
}

overwrite = True
num_partitions = 8  # Adjust based on your system

for file, set_type_value in data_files.items():
    file_path = os.path.join(DATA_FOLDER, file)
    
    # Read CSV file
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Add 'set_type' column
    df = df.withColumn('set_type', lit(set_type_value))
    
    # Repartition DataFrame
    df = df.repartition(num_partitions)
    
    # Show first 5 rows for verification
    df.show(5)
    
    # Determine write mode
    write_mode = 'overwrite' if overwrite else 'append'
    
    # Write DataFrame to PostgreSQL with batch size
    df.write.jdbc(
        url=JDBC_URL,
        table="mqtt",
        mode=write_mode,
        properties=DB_PROPERTIES,
        batchsize=10000
    )
    
    # Set overwrite to False after the first write
    if overwrite:
        overwrite = False

# Stop Spark session
spark.stop()
```

## Adjusting Spark Configurations: Static and Dynamic
**Static Configurations**
Cannot be changed after initialization and must use spark.stop() and start a new SparkSession
- **Memory Settings**:  spark.driver.memory, spark.executor.memory
- **Core Settings**: spark.executor.cores, spark.driver.cores
- **Application Name**: spark.app.name
- **JAR Files**: spark.jars

**Dynamic Configurations**
Certain SQL settings, Log levels, and Dynamic resource allocation
```python
# Check current setting
current_partitions = spark.conf.get("spark.sql.shuffle.partitions")
print(f"Current shuffle partitions: {current_partitions}")

# Adjust shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "8")  # Adjust the number as needed

# Verify the change
updated_partitions = spark.conf.get("spark.sql.shuffle.partitions")
print(f"Updated shuffle partitions: {updated_partitions}")

# Increase broadcast timeout to 1 hour
spark.conf.set("spark.sql.broadcastTimeout", "3600")

# Log levels
spark.sparkContext.setLogLevel("ERROR")  # Options: ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE, WARN
```

### Spark Configuration Hierarchy
Order of Precedence:
- **Runtime Configurations**: Settings made using spark.conf.set() have the highest priority for dynamic settings.
- **SparkSession Builder Configurations**: Settings provided when building the SparkSession.
- **spark-defaults.conf**: Cluster-wide default configurations.
- **Built-in Defaults**: Spark's internal default settings.

#### Examples of Configurations That Cannot Be Changed at Runtime
- spark.driver.memory
- spark.executor.memory
- spark.executor.instances
- spark.executor.cores
- spark.app.name
- spark.driver.extraClassPath
- spark.executor.extraClassPath

#### Examples of Configurations That Can Be Changed at Runtime
- spark.sql.shuffle.partitions
- spark.sql.autoBroadcastJoinThreshold
- spark.sql.broadcastTimeout
- spark.sql.adaptive.enabled
- spark.executor.heartbeatInterval
