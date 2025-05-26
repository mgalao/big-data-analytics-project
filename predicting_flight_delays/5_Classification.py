# Databricks notebook source
# MAGIC %md
# MAGIC # 5. Classification

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Libraries

# COMMAND ----------

from pyspark.sql.functions import lit, col, desc, to_date, length, count, when, hour, dayofweek, round, explode, lower, udf, mean, stddev, min, max, coalesce, row_number, monotonically_increasing_id, floor, round as spark_round
from pyspark.sql.types import ArrayType, StringType, FloatType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, BucketedRandomProjectionLSH
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.impute import KNNImputer
from functools import reduce

import re

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, PCA
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
import seaborn as sns

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("Classification").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Dfs

# COMMAND ----------

# File location and type
cluster_mapping_file_location = "/FileStore/tables/cluster_mapping.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
cluster_mapping_df = (
    spark.read.format(file_type)
    .option("inferSchema", infer_schema)
    .option("header", first_row_is_header)
    .option("sep", delimiter)
    .option("quote", '"') # To handle commas correctly
    .option("escape", '"')
    .option("multiLine", "true")  # Allow fields to span multiple lines (helps with complex quoted fields)
    .option("mode", "PERMISSIVE") # Avoid failing on corrupt records
    .load(cluster_mapping_file_location)
)

# Display result
display(cluster_mapping_df)

# COMMAND ----------

# Import train, val and test
train_df = spark.read.format("delta").load("/dbfs/FileStore/tables/train_df")
val_df = spark.read.format("delta").load("/dbfs/FileStore/tables/val_df")
test_df = spark.read.format("delta").load("/dbfs/FileStore/tables/test_df")

# Display train_df
display(train_df)

# COMMAND ----------

train_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Classification Target

# COMMAND ----------

# List of delay columns
delay_cols = [
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY"
]

# COMMAND ----------

from pyspark.sql.functions import col

# Create a condition to identifiy if all delay columns are equal to 0
all_zero_condition = reduce(lambda a, b: a & b, [(col(c) == 0) for c in delay_cols])

# Filter rows where all delay columns are 0
all_zero_delays_df = train_df.filter(all_zero_condition)

# Count them
zero_delay_count = all_zero_delays_df.count()

# Show count and display rows
print(f"Number of rows where all delay causes are zero: {zero_delay_count}")
all_zero_delays_df.select(delay_cols).display()

# COMMAND ----------

from pyspark.sql.functions import greatest, col, when
from functools import reduce

# Compute max delay
max_delay = greatest(*[col(c) for c in delay_cols])

# Create 1/0 indicator columns for each delay column if it's equal to max delay
indicator_cols = [(col(c) == max_delay).cast("int") for c in delay_cols]

# Sum how many match the max (potential ties)
tie_count_col = reduce(lambda a, b: a + b, indicator_cols)

# Add tie count column
cleaned_train_df_with_ties = train_df.withColumn("tie_count", tie_count_col)

# Filter and count rows with ties (more than one delay equals the max)
tie_rows_df = cleaned_train_df_with_ties.filter(col("tie_count") > 1)
tie_count = tie_rows_df.count()

print(f"Number of rows with ties in delay causes: {tie_count}")

# Show tie rows
tie_rows_df.select(delay_cols + ["tie_count"]).display()

# COMMAND ----------

from pyspark.sql.functions import when, col, greatest
from functools import reduce

# Define priority mapping
priority_order = [
    ("AIR_SYSTEM_DELAY", "AIR_SYSTEM"), # Delays caused by air traffic control or system-wide problems. Usually critical.
    ("SECURITY_DELAY", "SECURITY"), # Delays related to security checks or incidents. Also very important.
    ("AIRLINE_DELAY", "AIRLINE"), # Delays caused by airline operations, like crew or maintenance.
    ("LATE_AIRCRAFT_DELAY", "LATE_AIRCRAFT"), # Delay caused by the late arrival of the aircraft from a previous flight.
    ("WEATHER_DELAY", "WEATHER") # Weather-related delays, generally unpredictable and outside control.
]

max_delay = greatest(*[col(c) for c in delay_cols])
total_delay = reduce(lambda a, b: a + b, [col(c) for c in delay_cols])

# Assign function
def add_target(df):
    df = df.withColumn("MAX_DELAY", max_delay)
    df = df.withColumn("TOTAL_DELAY", total_delay)
    
    expr = None
    for delay_col, label in priority_order:
        condition = (col(delay_col) == col("MAX_DELAY"))
        expr = when(condition, label) if expr is None else expr.when(condition, label)
    
    expr = when(col("TOTAL_DELAY") == 0, "NO_DELAY").otherwise(expr)
    df = df.withColumn("PRIMARY_DELAY_CAUSE", expr)
    
    return df.drop("MAX_DELAY", "TOTAL_DELAY")

# Apply
cleaned_train_df = add_target(train_df)
cleaned_val_df = add_target(val_df)
cleaned_test_df = add_target(test_df)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# Create indexer and fit on train
target_indexer = StringIndexer(inputCol="PRIMARY_DELAY_CAUSE", outputCol="PRIMARY_DELAY_CAUSE_index")
target_indexer_model = target_indexer.fit(cleaned_train_df)

# Transform all datasets using the same fitted model
cleaned_train_df = target_indexer_model.transform(cleaned_train_df)
cleaned_val_df = target_indexer_model.transform(cleaned_val_df)
cleaned_test_df = target_indexer_model.transform(cleaned_test_df)

# COMMAND ----------

# # Keep target index and rename it for simplicity
# cleaned_train_df = cleaned_train_df.drop("PRIMARY_DELAY_CAUSE") \
#     .withColumnRenamed("PRIMARY_DELAY_CAUSE_index", "PRIMARY_DELAY_CAUSE")

# cleaned_val_df = cleaned_val_df.drop("PRIMARY_DELAY_CAUSE") \
#     .withColumnRenamed("PRIMARY_DELAY_CAUSE_index", "PRIMARY_DELAY_CAUSE")

# cleaned_test_df = cleaned_test_df.drop("PRIMARY_DELAY_CAUSE") \
#     .withColumnRenamed("PRIMARY_DELAY_CAUSE_index", "PRIMARY_DELAY_CAUSE")

# COMMAND ----------

cleaned_train_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Imbalance

# COMMAND ----------

# Count occurrences of each class in the target column
class_distribution = cleaned_train_df.groupBy("PRIMARY_DELAY_CAUSE").count().orderBy("count", ascending=False)

# Show the distribution
class_distribution.display()

# COMMAND ----------

# MAGIC %md
# MAGIC **Conclusion:** The dataset is imbalanced, with the WEATHER and SECURITY classes significantly underrepresented compared to others.

# COMMAND ----------

from pyspark.sql.functions import col, lit
from pyspark.sql import Window
import pyspark.sql.functions as F

# Get total number of samples
total = cleaned_train_df.count()

# Calculate class frequencies
class_counts = cleaned_train_df.groupBy("PRIMARY_DELAY_CAUSE").count()

# Compute inverse frequency as weights
weights_df = class_counts.withColumn("weight", round(lit(total) / col("count"), 4))

# Join weights back to original DataFrame
cleaned_train_df = cleaned_train_df.join(weights_df.select("PRIMARY_DELAY_CAUSE", "weight"), on="PRIMARY_DELAY_CAUSE", how="left")

weights_df.display()
cleaned_train_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Drop Features

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We aim to predict **ARRIVAL_DELAY** for flights after departure has already occurred. This means that we can safely use any features that become available after the flight takes off.
# MAGIC
# MAGIC However, to avoid data leakage, we must still exclude variables that contain or imply the final arrival delay itself, or any information that can only be known after landing.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Features to Exclude (Post Arrival Information or Leakage Risk)
# MAGIC
# MAGIC | Feature                | Reason for Exclusion                                                                 |
# MAGIC |------------------------|---------------------------------------------------------------------------------------|
# MAGIC | ARRIVAL_DELAY        | Target variable — cannot be used as input.                                           |
# MAGIC | ARRIVAL_TIME         | Actual arrival time — directly reveals the target.                                   |
# MAGIC | TAXI_IN              | Time spent taxiing after landing — only known after arrival.                         |
# MAGIC | WHEELS_ON            | Time the aircraft touched the runway — post-flight metric.                           |
# MAGIC | TOTAL_KNOWN_DELAY    | Sum of known delay components — often derived post-hoc.                              |
# MAGIC | AIR_SYSTEM_DELAY     | Delay cause — typically logged after analysis.                                       |
# MAGIC | SECURITY_DELAY       | Same — reported post-flight.                                                         |
# MAGIC | AIRLINE_DELAY        | Same — requires post-event information.                                              |
# MAGIC | LATE_AIRCRAFT_DELAY  | Same — only known after tracking inbound aircraft.                                   |
# MAGIC | WEATHER_DELAY        | Same — unless using real-time weather feed, it's a post-analysis variable.           |
# MAGIC | ELAPSED_TIME         | Total actual time of flight — includes arrival info.                                 |
# MAGIC | CANCELLED            | Indicates flight didn’t operate — not relevant when flight has already departed.     |
# MAGIC | DIVERTED             | Only known after the flight ends — may distort predictions if included.              |
# MAGIC | AIR_TIME             | Only known after the flight ends — reflects the actual flying duration from WHEELS_OFF to WHEELS_ON             |

# COMMAND ----------

# List of columns to drop
cols_to_drop = [
    "ARRIVAL_DELAY",
    "ARRIVAL_TIME",
    "TAXI_IN",
    "WHEELS_ON",
    "TOTAL_KNOWN_DELAY",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
    "ELAPSED_TIME",
    "CANCELLED",
    "DIVERTED",
    "AIR_TIME"
]

# Drop from DataFrames
cleaned_train_df = cleaned_train_df.drop(*cols_to_drop)
cleaned_val_df = cleaned_val_df.drop(*cols_to_drop)
cleaned_test_df = cleaned_test_df.drop(*cols_to_drop)

# COMMAND ----------

display(cleaned_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Scaling

# COMMAND ----------

from pyspark.sql.types import NumericType, StringType
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import json

# Initialize lists
categorical_columns = []
numerical_columns = []

# Target column
target_column = "PRIMARY_DELAY_CAUSE_index"

# Inspect schema and classify columns (excluding target)
for field in cleaned_train_df.schema.fields:
    if field.name in [target_column, "PRIMARY_DELAY_CAUSE", "weight"]:
        continue  # Skip the target and weight columns
    if isinstance(field.dataType, NumericType):
        numerical_columns.append(field.name)
        cleaned_train_df = cleaned_train_df.withColumn(field.name, col(field.name).cast("float"))
    else:
        categorical_columns.append(field.name)
        cleaned_train_df = cleaned_train_df.withColumn(field.name, col(field.name).cast(StringType()))

# Cast val and test to match train
for df_name, df in [("cleaned_val_df", cleaned_val_df), ("cleaned_test_df", cleaned_test_df)]:
    for col_name in numerical_columns:
        df = df.withColumn(col_name, col(col_name).cast("float"))
    for col_name in categorical_columns:
        df = df.withColumn(col_name, col(col_name).cast(StringType()))
    
    # Update the original reference
    if df_name == "cleaned_val_df":
        cleaned_val_df = df
    else:
        cleaned_test_df = df

# Add encoded features manually
encoded_columns = [
    "SEASON_vec", 
    "SCHEDULED_DEPARTURE_PERIOD_vec",
    # "AIRLINE_freq",
    # "ORIGIN_AIRPORT_CLEAN_freq",
    # "DESTINATION_AIRPORT_CLEAN_freq",
    # "ROUTE_freq"
]

def json_to_sparse_vector(s):
    from pyspark.ml.linalg import SparseVector
    if s is None or s.strip() == "":
        return SparseVector(0, [], [])
    try:
        obj = json.loads(s)
        if obj.get("vectorType") == "sparse":
            length = obj.get("length")
            indices = obj.get("indices", [])
            values = obj.get("values", [])
            return SparseVector(length, indices, values)
        else:
            values = obj.get("values", [])
            return SparseVector(len(values), list(range(len(values))), values)
    except Exception:
        return SparseVector(0, [], [])

json_to_sparse_vector_udf = udf(json_to_sparse_vector, VectorUDT())

# Apply the UDF on train, val, test datasets for the sparse encoded columns
for col_name in encoded_columns:
    cleaned_train_df = cleaned_train_df.withColumn(col_name, json_to_sparse_vector_udf(col_name))
    cleaned_val_df = cleaned_val_df.withColumn(col_name, json_to_sparse_vector_udf(col_name))
    cleaned_test_df = cleaned_test_df.withColumn(col_name, json_to_sparse_vector_udf(col_name))

# Final feature list
final_numerical_columns = numerical_columns + encoded_columns

# Output the lists
print("Categorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)
print("Final numerical columns:", final_numerical_columns)

# COMMAND ----------

# Assemble numerical features
assembler = VectorAssembler(
    inputCols=final_numerical_columns,
    outputCol="final_numerical_columns"
)

train_assembled_df = assembler.transform(cleaned_train_df)
val_assembled_df = assembler.transform(cleaned_val_df)
test_assembled_df = assembler.transform(cleaned_test_df)

# Fit scaler on train and apply to all
scaler = StandardScaler(
    inputCol="final_numerical_columns",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(train_assembled_df)
train_scaled_df = scaler_model.transform(train_assembled_df)
val_scaled_df = scaler_model.transform(val_assembled_df)
test_scaled_df = scaler_model.transform(test_assembled_df)

# Check if the datasets have the "scaled_numerical_features" column
train_scaled_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Create index Column

# COMMAND ----------

from pyspark.sql.functions import row_number, lit
from pyspark.sql.window import Window

# Define a dummy window specification (no particular order)
window_spec = Window.orderBy(lit(1))

# Add index column starting from 1
train_scaled_df = train_scaled_df.withColumn("index", row_number().over(window_spec))
val_scaled_df = val_scaled_df.withColumn("index", row_number().over(window_spec))
test_scaled_df = test_scaled_df.withColumn("index", row_number().over(window_spec))

# COMMAND ----------

train_scaled_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Cluster Assignment

# COMMAND ----------

# Join on 'index' to bring the 'cluster' column into train_df
train_cluster_df = train_scaled_df.join(cluster_mapping_df, on="index", how="left")
# val_df = val_df.join(cluster_mapping_df, on="index", how="left")
# test_df = test_df.join(cluster_mapping_df, on="index", how="left")

# Remove this later!!!
import pyspark.sql.functions as F
from pyspark.sql.functions import floor, rand

# Number of clusters
k = 6

# Assign random clusters to val_df and test_df
val_cluster_df = val_scaled_df.withColumn("cluster", floor(rand(seed=42) * k).cast("int"))
test_cluster_df = test_scaled_df.withColumn("cluster", floor(rand(seed=42) * k).cast("int"))

train_cluster_df.display()

# COMMAND ----------

# Check clusters in train
train_cluster_df.select("cluster").distinct().orderBy("cluster").show()
val_cluster_df.select("cluster").distinct().orderBy("cluster").show()
test_cluster_df.select("cluster").distinct().orderBy("cluster").show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Split Dfs by Cluster

# COMMAND ----------

# Get distinct cluster IDs from the mapping
cluster_ids = sorted(
    cluster_mapping_df.select("cluster").distinct().rdd.flatMap(lambda x: x).collect()
)

# Create a dictionary to hold cluster-specific splits
train_dfs_by_cluster = {}
val_dfs_by_cluster = {}
test_dfs_by_cluster = {}

# Split each DataFrame by cluster
for cluster_id in cluster_ids:
    train_dfs_by_cluster[cluster_id] = train_cluster_df.filter(col("cluster") == cluster_id).cache()
    val_dfs_by_cluster[cluster_id] = val_cluster_df.filter(col("cluster") == cluster_id).cache()
    test_dfs_by_cluster[cluster_id] = test_cluster_df.filter(col("cluster") == cluster_id).cache()

# COMMAND ----------

train_dfs_by_cluster

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spearman Correlation Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features for Clustering Flights
# MAGIC
# MAGIC #### Numerical Features
# MAGIC
# MAGIC | Feature               | Description                                          |
# MAGIC |-----------------------|------------------------------------------------------|
# MAGIC | `DISTANCE`            | Distance between origin and destination airports     |
# MAGIC | `TAXI_OUT`            | Time spent taxiing before takeoff (congestion proxy) |
# MAGIC | `DEPARTURE_TIME`      | Actual departure time                                |
# MAGIC | `WHEELS_OFF`          | Time when aircraft took off                          |
# MAGIC
# MAGIC #### Categorical Features
# MAGIC
# MAGIC | Feature                     | Description                                          |
# MAGIC |-----------------------------|------------------------------------------------------|
# MAGIC | `DAY_OF_WEEK`               | Day of the week (1 = Monday, ..., 7 = Sunday)        |
# MAGIC | `AIRLINE`                   | Airline operating the flight                         |
# MAGIC | `ORIGIN_AIRPORT`            | Code of the departure airport                        |
# MAGIC | `DESTINATION_AIRPORT`       | Code of the arrival airport                          |
# MAGIC | `ROUTE`                     | Combination of origin and destination                |
# MAGIC | `SEASON`                    | Season of the year      |
# MAGIC | `SCHEDULED_DEPARTURE_PERIOD`| Time-of-day label   |
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Loop through each cluster
for cluster_id, cluster_df in train_dfs_by_cluster.items():
    print(f"\n--- Cluster {cluster_id} ---")

    # Identify non-binary numerical features (for this cluster only)
    non_binary_numerical = [
        col_name for col_name in numerical_columns
        if cluster_df.select(col_name).distinct().count() > 2
    ]

    # Assemble features
    assembler = VectorAssembler(inputCols=non_binary_numerical, outputCol="non_binary_features")
    assembled = assembler.transform(cluster_df)

    # Compute Spearman correlation
    corr_matrix = Correlation.corr(assembled, "non_binary_features", method="spearman").collect()[0][0]
    corr_array = corr_matrix.toArray()

    # Create pandas DataFrame for visualization
    corr_df = pd.DataFrame(corr_array, columns=non_binary_numerical, index=non_binary_numerical)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(f"Cluster {cluster_id} - Spearman Correlation Matrix")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.stat import Correlation
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Identify non-binary numerical features
# non_binary_numerical = [col for col in numerical_columns if cleaned_train_df.select(col).distinct().count() > 2]

# # Assemble vector column only with non-binary numerical features
# assembler = VectorAssembler(inputCols=non_binary_numerical, outputCol="non_binary_features")
# train_non_binary = assembler.transform(train_scaled_df)

# # Compute Pearson correlation
# correlation_matrix = Correlation.corr(train_non_binary, "non_binary_features", method="spearman").collect()[0][0]
# corr_array = correlation_matrix.toArray()

# # Create and plot heatmap
# corr_df = pd.DataFrame(corr_array, columns=non_binary_numerical, index=non_binary_numerical)

# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", square=True)
# plt.title("Correlation Matrix (Non-binary Numerical Features)")
# plt.tight_layout()
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - WHEELS_OFF_min and DEPARTURE_TIME_min are highly correlated. To avoid redundancy in the feature space, we removed WHEELS_OFF_min.
# MAGIC - Although DEPARTURE_TIME_min and SCHEDULED_DEPARTURE_min also show a strong correlation (ρ = 0.83), we chose to retain both. This allows us to capture potential delays and their impact on flight behavior, which can be informative for clustering.
# MAGIC

# COMMAND ----------

# for cat in categorical_columns:
#     for num in numerical_columns:
#         print(f"\n Mean {num} by {cat}:")
#         cleaned_train_df.groupBy(cat).avg(num).orderBy(cat).show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Day of the Week
# MAGIC - Flights on Saturday tend to cover the longest distances, while those on Wednesday are the shortest.
# MAGIC - Taxi-out times are slightly shorter on weekends.
# MAGIC - Scheduled and actual departure times are generally later on Sundays.
# MAGIC - IS_WEEKEND is redundant with this feature.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Airline 
# MAGIC - Airlines such as VX, UA, and AS operate longer routes on average, while MQ, EV, and OO operate shorter flights.
# MAGIC - Taxi-out times vary: DL and US experience higher values, while HA and WN have the shortest.
# MAGIC - Departure times are later for WN and B6.
# MAGIC - Weekend proportions are slightly higher for NK and F9.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Airport 
# MAGIC - Certain airports (ANC, ADK) are associated with significantly longer routes.
# MAGIC - Airports like ATL, JFK, ORD have higher average taxi-out times, reflecting busier traffic and congestion.
# MAGIC - Departure and arrival times vary substantially depending on the origin/destination.
# MAGIC
# MAGIC **Note**: ORIGIN_AIRPORT_CLEAN and DESTINATION_AIRPORT_CLEAN should be retained, as they explain a lot of operational variability.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC #### Scheduled Time Periods
# MAGIC - Flights in the Late Night period have the longest distances and earliest clock times (e.g., 12:30 AM).
# MAGIC - Afternoon and Evening flights tend to occur later and with higher taxi-out times.
# MAGIC - Early Morning flights face more congestion, leading to longer taxi-out times.
# MAGIC
# MAGIC **Note**: SCHEDULED_DEPARTURE_PERIOD captures time-of-day effects cleanly. It may serve as a compact alternative to raw time variables.
# MAGIC
# MAGIC
# MAGIC ## Feature Selection Summary for Clustering
# MAGIC
# MAGIC ### Numerical Features
# MAGIC
# MAGIC | Feature                   | Keep? | Reason                                                                 |
# MAGIC |---------------------------|-------|------------------------------------------------------------------------|
# MAGIC | `DISTANCE`               | Yes |    |
# MAGIC | `TAXI_OUT`               | Yes |    |
# MAGIC | `DEPARTURE_TIME_min`     | Yes |    |
# MAGIC | `SCHEDULED_DEPARTURE_min`| No  | Redundant with SCHEDULED_DEPARTURE_PERIOD |
# MAGIC | `SCHEDULED_ARRIVAL_min`  | No  | Adds little; not as informative for clustering as departure features. |
# MAGIC | `WHEELS_OFF_min`         | No  | Highly correlated with DEPARTURE_TIME_min.                          |
# MAGIC | `IS_WEEKEND`             | No  | Redundant with DAY_OF_WEEK              |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Categorical Features
# MAGIC
# MAGIC | Feature                       | Keep? | Reason                                                                   |
# MAGIC |------------------------------|-------|--------------------------------------------------------------------------|
# MAGIC | `DAY_OF_WEEK`                | Yes |     |
# MAGIC | `AIRLINE`                    | Yes |     |
# MAGIC | `ORIGIN_AIRPORT_CLEAN`       | Yes |     |
# MAGIC | `DESTINATION_AIRPORT_CLEAN`  | Yes |     |
# MAGIC | `SEASON`                     | Yes |     |
# MAGIC | `SCHEDULED_DEPARTURE_PERIOD`| Yes |      |
# MAGIC | `ROUTE`                      | No  | Redundant with origin + destination; introduces unnecessary sparsity.    |
# MAGIC
# MAGIC

# COMMAND ----------

# Define exclusion lists per cluster
excluded_columns_per_cluster = {
    0: ['DEPARTURE_DELAY', 'ROUTE'],
    1: ['DEPARTURE_DELAY', 'IS_WEEKEND'],
    2: ['DEPARTURE_DELAY'],
    3: ['DEPARTURE_DELAY'],
    4: ['DEPARTURE_DELAY'],
    5: ['DEPARTURE_DELAY'],
}

# Dictionary to store selected features per cluster
selected_features_spearman_by_cluster = {}

for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")

    cluster_df = train_dfs_by_cluster[cluster_id]
    cluster_numerical_columns = [col for col in final_numerical_columns if col in cluster_df.columns]

    # Get exclusion list for the current cluster or default empty list
    excluded_columns = excluded_columns_per_cluster.get(cluster_id)

    selected = [col for col in cluster_numerical_columns if col not in excluded_columns]

    selected_features_spearman_by_cluster[cluster_id] = selected

    print("Excluded features:")
    print(excluded_columns)

    print("Selected features:")
    print(selected)

# COMMAND ----------

# # Columns to exclude
# excluded_columns = [
#     'DEPARTURE_DELAY',
#     # 'SCHEDULED_DEPARTURE_min',
#     # 'SCHEDULED_ARRIVAL_min',
#     # 'WHEELS_OFF_min',
#     # 'IS_WEEKEND',
#     # 'ROUTE'
# ]

# # Filter the list
# selected_features_spearman = [col for col in final_numerical_columns if col not in excluded_columns]

# # Print results
# print("Spearman - excluded features:")
# print(excluded_columns)

# print("\nSpearman - selected features:")
# print(selected_features_spearman)

# COMMAND ----------

# MAGIC %md
# MAGIC ## RFE selection

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

# Parameters
target_num_features = 5  # number of features to select

# Dictionaries to store results per cluster
selected_features_rfe_by_cluster = {}
excluded_features_rfe_by_cluster = {}

for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")

    # Get the train subset for this cluster
    cluster_train_df = train_dfs_by_cluster[cluster_id]

    # Start with full features list (ensure they exist in cluster df)
    features = [f for f in final_numerical_columns if f in cluster_train_df.columns]
    excluded_features_rfe = []

    # RFE loop for this cluster
    while len(features) > target_num_features:
        assembler = VectorAssembler(inputCols=features, outputCol="features_temp")
        assembled_df = assembler.transform(cluster_train_df).select("features_temp", "PRIMARY_DELAY_CAUSE_index")

        rf = RandomForestRegressor(featuresCol="features_temp", labelCol="PRIMARY_DELAY_CAUSE_index", seed=42)
        model = rf.fit(assembled_df)

        importances = model.featureImportances.toArray()
        feature_importance_dict = dict(zip(features, importances))

        least_important = sorted(feature_importance_dict.items(), key=lambda x: x[1])[0][0]

        print(f"Removed: {least_important} (importance: {feature_importance_dict[least_important]:.6f})")
        features.remove(least_important)
        excluded_features_rfe.append(least_important)

    selected_features_rfe_by_cluster[cluster_id] = features
    excluded_features_rfe_by_cluster[cluster_id] = excluded_features_rfe

    print(f"Cluster {cluster_id} - Selected features: {features}")

# COMMAND ----------

# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.regression import RandomForestRegressor
# import json

# # Create features
# features = final_numerical_columns.copy()

# # Track removed features
# excluded_features_rfe = []

# # Number of features to select
# target_num_features = 5  # change this as needed

# # RFE-like feature elimination
# while len(features) > target_num_features:
#     # Assemble features
#     assembler = VectorAssembler(inputCols=features, outputCol="features_temp")
#     assembled_df = assembler.transform(train_scaled_df).select("features_temp", "PRIMARY_DELAY_CAUSE_index")
    
#     # Fit Random Forest
#     rf = RandomForestRegressor(featuresCol="features_temp", labelCol="PRIMARY_DELAY_CAUSE_index", seed=42)
#     model = rf.fit(assembled_df)
    
#     # Get importances and remove the least important
#     importances = model.featureImportances.toArray()
#     feature_importance_dict = dict(zip(features, importances))
#     least_important = sorted(feature_importance_dict.items(), key=lambda x: x[1])[0][0]
    
#     print(f"Removed: {least_important} (importance: {feature_importance_dict[least_important]:.6f})")
#     features.remove(least_important)
#     excluded_features_rfe.append(least_important)

# # Save selected features
# selected_features_rfe = features

# # Print results
# print("\nRFE - excluded features:")
# print(excluded_features_rfe)

# print("\nRFE - selected features:")
# print(selected_features_rfe)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DT Selection

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql.functions import lit
from pyspark.sql import Row

# Store results per cluster
selected_features_dt_by_cluster = {}
excluded_features_dt_by_cluster = {}

for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")

    # Get cluster-specific train data
    cluster_train_df = train_dfs_by_cluster[cluster_id]

    # Copy features list
    features = [f for f in final_numerical_columns if f in cluster_train_df.columns]

    # Assemble features
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    train_vectorized = assembler.transform(cluster_train_df).select("features", "PRIMARY_DELAY_CAUSE_index")

    # Train Decision Tree Regressor
    dt = DecisionTreeRegressor(featuresCol="features", labelCol="PRIMARY_DELAY_CAUSE_index", seed=42)
    dt_model = dt.fit(train_vectorized)

    # Extract feature importances
    importances = dt_model.featureImportances.toArray()

    # Convert to Spark DataFrame
    importance_rows = [Row(feature=feature, importance=float(score)) for feature, score in zip(features, importances)]
    importance_df = spark.createDataFrame(importance_rows)

    # Filter non-zero and sort descending
    selected_by_tree_df = importance_df.filter("importance > 0").orderBy("importance", ascending=False)

    # Collect selected features into a list
    selected_features_dt = [row["feature"] for row in selected_by_tree_df.collect()]

    # Compute excluded features
    excluded_features_dt = [f for f in features if f not in selected_features_dt]

    # Store per cluster
    selected_features_dt_by_cluster[cluster_id] = selected_features_dt
    excluded_features_dt_by_cluster[cluster_id] = excluded_features_dt

    # Print results
    print(f"Cluster {cluster_id} - DT excluded features: {excluded_features_dt}")
    print(f"Cluster {cluster_id} - DT selected features: {selected_features_dt}")

# COMMAND ----------

# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.regression import DecisionTreeRegressor
# from pyspark.sql.functions import lit
# from pyspark.sql import Row

# # Create features
# features = final_numerical_columns.copy()

# # Assemble features
# assembler = VectorAssembler(inputCols=features, outputCol="features")
# train_vectorized = assembler.transform(train_scaled_df).select("features", "PRIMARY_DELAY_CAUSE_index")

# # Train Decision Tree
# dt = DecisionTreeRegressor(featuresCol="features", labelCol="PRIMARY_DELAY_CAUSE_index", seed=42)
# dt_model = dt.fit(train_vectorized)

# # Extract feature importances
# importances = dt_model.featureImportances.toArray()

# # Convert to Spark DataFrame
# importance_rows = [Row(feature=feature, importance=float(score)) for feature, score in zip(features, importances)]
# importance_df = spark.createDataFrame(importance_rows)

# # Filter non-zero and sort
# selected_by_tree_df = importance_df.filter("importance > 0").orderBy("importance", ascending=False)

# # Collect selected features into a list
# selected_features_dt = [row["feature"] for row in selected_by_tree_df.collect()]

# # Compute excluded features
# excluded_features_dt = [f for f in features if f not in selected_features_dt]

# # Print results
# print("DT - excluded features:")
# print(excluded_features_dt)

# print("\nDT - selected features:")
# print(selected_features_dt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lasso Selection

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

selected_features_lasso_by_cluster = {}
excluded_features_lasso_by_cluster = {}

for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")

    # Get cluster-specific train data
    cluster_train_df = train_dfs_by_cluster[cluster_id]

    # Check features actually present in this cluster df
    features = [f for f in final_numerical_columns if f in cluster_train_df.columns]

    # Assemble features vector
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    train_data = assembler.transform(cluster_train_df).select("features", "PRIMARY_DELAY_CAUSE_index")

    # Rename label column for LinearRegression
    train_data = train_data.withColumnRenamed("PRIMARY_DELAY_CAUSE_index", "label")

    # Initialize Lasso (elasticNetParam=1.0 for Lasso)
    lasso = LinearRegression(featuresCol="features", labelCol="label", elasticNetParam=1.0, regParam=0.1)

    # Fit model
    lasso_model = lasso.fit(train_data)

    # Get coefficients as array
    coefficients = lasso_model.coefficients.toArray()

    # Identify selected and excluded features based on coefficients
    selected_features = [f for f, coef in zip(features, coefficients) if coef != 0]
    excluded_features = [f for f, coef in zip(features, coefficients) if coef == 0]

    # Store results
    selected_features_lasso_by_cluster[cluster_id] = selected_features
    excluded_features_lasso_by_cluster[cluster_id] = excluded_features

    # Print for this cluster
    print(f"Cluster {cluster_id} - Lasso excluded features: {excluded_features}")
    print(f"Cluster {cluster_id} - Lasso selected features: {selected_features}")

# COMMAND ----------

# from pyspark.ml.regression import LinearRegression
# from pyspark.sql import Row

# # Prepare training data
# train_data = train_scaled_df.select("final_numerical_columns", "PRIMARY_DELAY_CAUSE_index") \
#     .withColumnRenamed("final_numerical_columns", "features") \
#     .withColumnRenamed("PRIMARY_DELAY_CAUSE_index", "label")

# # Initialize Lasso Regression
# lasso = LinearRegression(featuresCol="features", labelCol="label", elasticNetParam=1.0, regParam=0.1)

# # Fit the model
# lasso_model = lasso.fit(train_data)

# # Get coefficients and feature names
# coefficients = lasso_model.coefficients.toArray()
# feature_names = final_numerical_columns

# # Separate selected and excluded features
# selected_features_lasso = [feature for feature, coef in zip(feature_names, coefficients) if coef != 0]
# excluded_features_lasso = [feature for feature, coef in zip(feature_names, coefficients) if coef == 0]

# # Print results
# print("Lasso - excluded features:")
# print(excluded_features_lasso)

# print("\nLasso - selected features:")
# print(selected_features_lasso)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Majority Voting

# COMMAND ----------

from collections import Counter

majority_voted_features_by_cluster = {}
excluded_majority_features_by_cluster = {}

methods = [
    'spearman',
    'rfe',
    'dt',
    'lasso'
]

for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")

    # Collect features selected by each method for this cluster
    feature_lists = [
        selected_features_spearman_by_cluster.get(cluster_id),
        selected_features_rfe_by_cluster.get(cluster_id),
        selected_features_dt_by_cluster.get(cluster_id),
        selected_features_lasso_by_cluster.get(cluster_id)
    ]

    # Flatten and count occurrences
    all_selected = sum(feature_lists, [])
    feature_counts = Counter(all_selected)

    # Majority threshold (at least half)
    threshold = len(feature_lists) // 2

    majority_voted_features = [f for f, count in feature_counts.items() if count >= threshold]
    excluded_majority_features = [f for f in feature_counts if f not in majority_voted_features]

    # Store results
    majority_voted_features_by_cluster[cluster_id] = majority_voted_features
    excluded_majority_features_by_cluster[cluster_id] = excluded_majority_features

    # Print cluster results
    print(f"Cluster {cluster_id} - Majority Voting Selected Features:")
    print(majority_voted_features)
    print(f"Cluster {cluster_id} - Majority Voting Excluded Features:")
    print(excluded_majority_features)

# COMMAND ----------

# from collections import Counter

# feature_lists = [
#     selected_features_spearman,
#     selected_features_rfe,
#     selected_features_dt,
#     selected_features_lasso
# ]

# # Flatten all features into one list
# all_selected = sum(feature_lists, [])

# # Count occurrences of each feature
# feature_counts = Counter(all_selected)

# # Threshold: majority = at least half of the methods (rounded down)
# threshold = len(feature_lists) // 2

# # Features selected by majority voting
# majority_voted_features = [feature for feature, count in feature_counts.items() if count >= threshold]

# # Features excluded by majority voting
# excluded_majority_features = [feature for feature in feature_counts if feature not in majority_voted_features]

# # Print results
# print("Majority Voting - excluded features:")
# print(excluded_majority_features)

# print("\nMajority Voting - selected features:")
# print(majority_voted_features)

# COMMAND ----------

# MAGIC %md
# MAGIC # Classification

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import functions as F

# Assemble features
assembler = VectorAssembler(inputCols=majority_voted_features, outputCol="features")
train_prepared = assembler.transform(train_scaled_df)
val_prepared = assembler.transform(val_scaled_df)

# Train model
rf = RandomForestClassifier(
    featuresCol="features", 
    labelCol=target_column, 
    numTrees=100, 
    weightCol="weight", 
    seed=42
)
model = rf.fit(train_prepared)

# Predict
train_predictions = model.transform(train_prepared)
val_predictions = model.transform(val_prepared)

# COMMAND ----------

# Decode predictions for better interpretation
def decode_predictions(df):
    df = IndexToString(inputCol="prediction", outputCol="predicted_label", labels=target_indexer_model.labels).transform(df)
    df = IndexToString(inputCol=target_column, outputCol="true_label", labels=target_indexer_model.labels).transform(df)
    return df

train_decoded_predictions = decode_predictions(train_predictions)
val_decoded_predictions = decode_predictions(val_predictions)

# Show true vs predicted labels
val_decoded_predictions.select("true_label", "predicted_label").display()

# COMMAND ----------

# Function to compute macro F1
def compute_macro_f1(decoded_df):
    import pyspark.sql.functions as F

    # Compute confusion matrix
    cm = decoded_df.groupBy("true_label", "predicted_label").count()

    # Extract TP (where prediction == true)
    tp = cm.filter(F.col("true_label") == F.col("predicted_label")) \
           .select(F.col("true_label").alias("label"), F.col("count").alias("tp")).alias("tp")

    # Count true labels per class
    total_true = cm.groupBy("true_label").agg(F.sum("count").alias("true_total")) \
                   .select(F.col("true_label").alias("label"), "true_total").alias("true_total")

    # Count predicted labels per class
    total_pred = cm.groupBy("predicted_label").agg(F.sum("count").alias("pred_total")) \
                   .select(F.col("predicted_label").alias("label"), "pred_total").alias("total_pred")

    # Join all metrics on label
    metrics = total_true.join(tp, on="label", how="outer") \
                        .join(total_pred, on="label", how="outer") \
                        .select(
                            F.col("label"),
                            F.coalesce(F.col("tp.tp"), F.lit(0)).alias("tp"),
                            F.coalesce(F.col("true_total.true_total"), F.lit(0)).alias("true_total"),
                            F.coalesce(F.col("total_pred.pred_total"), F.lit(0)).alias("pred_total")
                        )

    # Compute precision, recall, F1 per class
    metrics = metrics.withColumn("precision", F.when(F.col("pred_total") != 0, F.col("tp") / F.col("pred_total")).otherwise(0))
    metrics = metrics.withColumn("recall", F.when(F.col("true_total") != 0, F.col("tp") / F.col("true_total")).otherwise(0))
    metrics = metrics.withColumn("f1", F.when(
        (F.col("precision") + F.col("recall")) > 0,
        2 * (F.col("precision") * F.col("recall")) / (F.col("precision") + F.col("recall"))
    ).otherwise(0))

    # Average F1 across classes
    macro_f1 = metrics.select(F.avg("f1")).first()[0]

    return macro_f1

# Standard evaluator for accuracy and weighted F1
evaluator_acc = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction", metricName="f1")

# Train metrics
train_acc = evaluator_acc.evaluate(train_predictions)
train_weighted_f1 = evaluator_f1.evaluate(train_predictions)
train_macro_f1 = compute_macro_f1(train_decoded_predictions)

# Validation metrics
val_acc = evaluator_acc.evaluate(val_predictions)
val_weighted_f1 = evaluator_f1.evaluate(val_predictions)
val_macro_f1 = compute_macro_f1(val_decoded_predictions)

# Results
print(f"Train Accuracy:       {train_acc:.4f}")
print(f"Train Weighted F1:    {train_weighted_f1:.4f}")
print(f"Train Macro F1:       {train_macro_f1:.4f}")

print(f"\nValidation Accuracy:  {val_acc:.4f}")
print(f"Validation Weighted F1: {val_weighted_f1:.4f}")
print(f"Validation Macro F1:    {val_macro_f1:.4f}")

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix DataFrame
confusion_df = val_decoded_predictions.groupBy("true_label", "predicted_label").count()

# Convert to Pandas DataFrame and pivot
confusion_pd = confusion_df.toPandas().pivot(index="true_label", columns="predicted_label", values="count").fillna(0)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_pd, annot=True, fmt=".0f", cmap="Blues", cbar=True)
plt.title("Confusion Matrix (Validation Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Export Dataframe

# COMMAND ----------

# clustered_df.write.mode("overwrite") \
#     .option("overwriteSchema", "true") \
#     .format("delta") \
#     .save("/dbfs/FileStore/tables/clustered_train_df")