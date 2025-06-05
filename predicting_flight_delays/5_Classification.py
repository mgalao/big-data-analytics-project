# Databricks notebook source
# MAGIC %md
# MAGIC # 5. Classification

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.1 Import Libraries

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
# MAGIC # 5.2 Import Dfs

# COMMAND ----------

# Import train, val and test
train_df = spark.read.format("delta").load("/dbfs/FileStore/tables/train_df")
val_df = spark.read.format("delta").load("/dbfs/FileStore/tables/val_df")
test_df = spark.read.format("delta").load("/dbfs/FileStore/tables/test_df")

# Display train_df
train_df.limit(10).display()

# COMMAND ----------

# File location and type
clust_map_train_df_file_loc = "/FileStore/tables/mapping_train_df.csv"
clust_map_val_df_file_loc = "/FileStore/tables/mapping_val_df.csv"
clust_map_test_df_file_loc = "/FileStore/tables/mapping_test_df.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
clust_map_train_df = (
    spark.read.format(file_type)
    .option("inferSchema", infer_schema)
    .option("header", first_row_is_header)
    .option("sep", delimiter)
    .option("quote", '"') # To handle commas correctly
    .option("escape", '"')
    .option("multiLine", "true")  # Allow fields to span multiple lines (helps with complex quoted fields)
    .option("mode", "PERMISSIVE") # Avoid failing on corrupt records
    .load(clust_map_train_df_file_loc)
)

clust_map_val_df = (
    spark.read.format(file_type)
    .option("inferSchema", infer_schema)
    .option("header", first_row_is_header)
    .option("sep", delimiter)
    .option("quote", '"') # To handle commas correctly
    .option("escape", '"')
    .option("multiLine", "true")  # Allow fields to span multiple lines (helps with complex quoted fields)
    .option("mode", "PERMISSIVE") # Avoid failing on corrupt records
    .load(clust_map_val_df_file_loc)
)

clust_map_test_df = (
    spark.read.format(file_type)
    .option("inferSchema", infer_schema)
    .option("header", first_row_is_header)
    .option("sep", delimiter)
    .option("quote", '"') # To handle commas correctly
    .option("escape", '"')
    .option("multiLine", "true")  # Allow fields to span multiple lines (helps with complex quoted fields)
    .option("mode", "PERMISSIVE") # Avoid failing on corrupt records
    .load(clust_map_test_df_file_loc)
)

# Display result
clust_map_train_df.limit(10).display()

# COMMAND ----------

train_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.3 Create Classification Target

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
all_zero_delays_df.select(delay_cols).limit(10).display()

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
tie_rows_df.select(delay_cols + ["tie_count"]).limit(10).display()

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

cleaned_train_df.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.4 Imbalance

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
cleaned_train_df.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.5 Drop Features

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
    "FLIGHT_NUMBER", # should be dropped in preproc
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
    "AIR_TIME",
    "TOTAL_FLIGHT_MIDNIGHT_min"
]

# Drop from DataFrames
cleaned_train_df = cleaned_train_df.drop(*cols_to_drop)
cleaned_val_df = cleaned_val_df.drop(*cols_to_drop)
cleaned_test_df = cleaned_test_df.drop(*cols_to_drop)

# COMMAND ----------

cleaned_train_df.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.6 Define Feature Lists

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
    if field.name in [target_column, "PRIMARY_DELAY_CAUSE", "index", "weight"]:
        continue  # Skip the target and weight columns
    if isinstance(field.dataType, NumericType):
        numerical_columns.append(field.name)
        cleaned_train_df = cleaned_train_df.withColumn(field.name, col(field.name).cast("float"))
    else:
        categorical_columns.append(field.name)
        cleaned_train_df = cleaned_train_df.withColumn(field.name, col(field.name).cast(StringType()))

# Output the lists
print("Categorical columns:", categorical_columns)
print("Numerical columns:", numerical_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.7 Fix Data Types

# COMMAND ----------

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

# Encoded features
encoded_columns = [
    'SEASON_vec',
    'SCHEDULED_DEPARTURE_PERIOD_vec'
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

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.8 Scaling

# COMMAND ----------

# Assemble numerical features
assembler = VectorAssembler(
    inputCols=numerical_columns,
    outputCol="numerical_vector"
)

train_assembled_df = assembler.transform(cleaned_train_df)
val_assembled_df = assembler.transform(cleaned_val_df)
test_assembled_df = assembler.transform(cleaned_test_df)

# Fit scaler on train and apply to all
scaler = StandardScaler(
    inputCol="numerical_vector",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(train_assembled_df)
train_scaled_df = scaler_model.transform(train_assembled_df)
val_scaled_df = scaler_model.transform(val_assembled_df)
test_scaled_df = scaler_model.transform(test_assembled_df)

# Check if the datasets have the "scaled_numerical_features" column
train_scaled_df.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.9 Cluster Assignment

# COMMAND ----------

# Join on 'index' to bring the 'cluster' column into train_df
train_cluster_df = train_scaled_df.join(clust_map_train_df, on="index", how="left")
val_cluster_df = val_scaled_df.join(clust_map_val_df, on="index", how="left")
test_cluster_df = test_scaled_df.join(clust_map_test_df, on="index", how="left")

train_cluster_df.limit(10).display()

# COMMAND ----------

# Check clusters in train
train_cluster_df.select("cluster").distinct().orderBy("cluster").show()
val_cluster_df.select("cluster").distinct().orderBy("cluster").show()
test_cluster_df.select("cluster").distinct().orderBy("cluster").show()

# COMMAND ----------

# Compare row counts for each dataset pair (with clusters, scaled)
datasets = [
    ("Train", train_cluster_df, train_scaled_df),
    ("Validation", val_cluster_df, val_scaled_df),
    ("Test", test_cluster_df, test_scaled_df)
]

for name, cluster_df, scaled_df in datasets:
    cluster_count = cluster_df.count()
    scaled_count = scaled_df.count()
    match_status = "✅ Match" if cluster_count == scaled_count else "❌ Mismatch"
    print(f"{name} Set → Cluster: {cluster_count}, Scaled: {scaled_count} → {match_status}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.10 Split Dfs by Cluster

# COMMAND ----------

# Get distinct cluster IDs from the mapping
cluster_ids = sorted(
    clust_map_train_df.select("cluster").distinct().rdd.flatMap(lambda x: x).collect()
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
# MAGIC # 5.11 Feature Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.11.1 Spearman Correlation Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.11.1.1 Before

# COMMAND ----------

# Identify non-binary numerical features
non_binary_numerical = [
    col_name for col_name in numerical_columns
    if cluster_df.select(col_name).distinct().count() > 2
]
print("Non-Binary Numerical columns:", non_binary_numerical)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Loop through each cluster
for cluster_id, cluster_df in train_dfs_by_cluster.items():
    print(f"\n--- Cluster {cluster_id} ---")

    # Final list of features (target first)
    features_for_corr = [target_column] + non_binary_numerical

    # Assemble features
    assembler = VectorAssembler(inputCols=features_for_corr, outputCol="features_with_target")
    assembled = assembler.transform(cluster_df)

    # Compute Spearman correlation
    corr_matrix = Correlation.corr(assembled, "features_with_target", method="spearman").collect()[0][0]
    corr_array = corr_matrix.toArray()

    # Create pandas DataFrame for visualization
    corr_df = pd.DataFrame(corr_array, columns=features_for_corr, index=features_for_corr)

    # Mask the upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_df, 
        mask=mask, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        square=True,
        vmin=0, vmax=1
    )
    plt.title(f"Cluster {cluster_id} - Spearman Correlation Matrix (Lower Triangle)")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.11.1.2 After

# COMMAND ----------

# MAGIC %md
# MAGIC Threshold between features (to exclude): >= |0.80|
# MAGIC
# MAGIC Threshold features with target (to exclude): <= |0.10|

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql.functions import col
import numpy as np

# Define exclusion lists per cluster
high_corr_features_per_cluster = {
    0: ['SCHEDULED_TIME', 'DEPARTURE_TIME_min', 'WHEELS_OFF_min'],
    1: ['SCHEDULED_TIME', 'DEPARTURE_TIME_min'],
    2: ['SCHEDULED_TIME', 'DEPARTURE_TIME_min', 'WHEELS_OFF_min'],
    3: ['SCHEDULED_TIME', 'DEPARTURE_TIME_min', 'WHEELS_OFF_min'],
    4: ['SCHEDULED_TIME', 'DEPARTURE_TIME_min', 'WHEELS_OFF_min'],
}

# Threshold for weak correlation with target
corr_threshold_with_target = 0.1

# Dictionary to store selected features per cluster
selected_features_spearman_by_cluster = {}

for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")

    cluster_df = train_dfs_by_cluster[cluster_id]

    # Get high-correlated features
    high_corr_features = high_corr_features_per_cluster.get(cluster_id, [])

    # Compute Spearman correlation
    features_to_test = [target_column] + non_binary_numerical
    assembler = VectorAssembler(inputCols=features_to_test, outputCol="features_with_target")
    assembled = assembler.transform(cluster_df)

    corr_matrix = Correlation.corr(assembled, "features_with_target", method="spearman").collect()[0][0]
    corr_array = corr_matrix.toArray()

    # Target is at index 0
    correlations_with_target = corr_array[0, 1:]  # skip self-correlation

    low_corr_features_with_target = [
        feature for feature, corr in zip(non_binary_numerical, correlations_with_target)
        if abs(corr) <= corr_threshold_with_target
    ]

    # Combine exclusions and deduplicate
    total_excluded = list(set(high_corr_features + low_corr_features_with_target))

    # Final selected features
    selected = [col_name for col_name in non_binary_numerical if col_name not in total_excluded]

    selected_features_spearman_by_cluster[cluster_id] = selected

    print("High-correlation exclusions:")
    print(high_corr_features)
    print("Low-correlation with target exclusions:")
    print(low_corr_features_with_target)
    print("Selected features:")
    print(selected)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Loop through each cluster
for cluster_id, cluster_df in train_dfs_by_cluster.items():
    print(f"\n--- Cluster {cluster_id} ---")

    # Final list of selected features (and target)
    features_for_corr = [target_column] + selected_features_spearman_by_cluster[cluster_id]

    # Assemble features
    assembler = VectorAssembler(inputCols=features_for_corr, outputCol="features_with_target")
    assembled = assembler.transform(cluster_df)

    # Compute Spearman correlation
    corr_matrix = Correlation.corr(assembled, "features_with_target", method="spearman").collect()[0][0]
    corr_array = corr_matrix.toArray()

    # Create pandas DataFrame for visualization
    corr_df = pd.DataFrame(corr_array, columns=features_for_corr, index=features_for_corr)

    # Mask the upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool))

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_df, 
        mask=mask, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        square=True,
        vmin=0, vmax=1
    )
    plt.title(f"Cluster {cluster_id} - Spearman Correlation Matrix (Lower Triangle)")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.11.2 RFE selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.11.2.1 Functions

# COMMAND ----------

# Function to decode predictions (add true_label and predicted_label columns)
def decode_predictions(df):
    df = IndexToString(inputCol="prediction", outputCol="predicted_label", labels=target_indexer_model.labels).transform(df)
    df = IndexToString(inputCol=target_column, outputCol="true_label", labels=target_indexer_model.labels).transform(df)
    return df

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.11.2.2 Choose Number of Features to Select

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
import matplotlib.pyplot as plt

for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")
    
    cluster_train_df = train_dfs_by_cluster[cluster_id]
    cluster_val_df = val_dfs_by_cluster[cluster_id]
    features = [f for f in numerical_columns if f in cluster_train_df.columns]

    features_history = []
    train_f1_history = []
    val_f1_history = []
    
    excluded_features = []

    while len(features) > 1:
        assembler = VectorAssembler(inputCols=features, outputCol="features_temp")
        
        # Assemble both train and val
        train_assembled = assembler.transform(cluster_train_df).select("features_temp", "PRIMARY_DELAY_CAUSE_index")
        val_assembled = assembler.transform(cluster_val_df).select("features_temp", "PRIMARY_DELAY_CAUSE_index")

        rf = RandomForestClassifier(featuresCol="features_temp", labelCol="PRIMARY_DELAY_CAUSE_index", seed=42)
        model = rf.fit(train_assembled)

        # Predict train and val
        train_preds = decode_predictions(model.transform(train_assembled))
        val_preds = decode_predictions(model.transform(val_assembled))

        # Compute metrics
        train_f1 = compute_macro_f1(train_preds)
        val_f1 = compute_macro_f1(val_preds)

        # Store history
        features_history.append(features.copy())
        train_f1_history.append(train_f1)
        val_f1_history.append(val_f1)

        # Remove least important
        importances = model.featureImportances.toArray()
        feature_importance_dict = dict(zip(features, importances))
        least_important = sorted(feature_importance_dict.items(), key=lambda x: x[1])[0][0]

        print(f"Removed: {least_important} (importance: {feature_importance_dict[least_important]:.6f}, F1-train: {train_f1:.4f}, F1-val: {val_f1:.4f})")
        features.remove(least_important)
        excluded_features.append(least_important)

    # Plot elbow
    plt.figure(figsize=(10, 6))
    plt.plot([len(f) for f in features_history], train_f1_history, label="Train F1", marker='o')
    plt.plot([len(f) for f in features_history], val_f1_history, label="Val F1", marker='s')
    plt.xlabel("Number of Features")
    plt.ylabel("F1 Macro")
    plt.title(f"Cluster {cluster_id} - F1 vs. Number of Features")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.11.2.3 RFE

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor

# Define number of features to select per cluster (based on previous elbows)
target_num_features_by_cluster = {
    0: 8,
    1: 4,
    2: 6,
    3: 5,
    4: 4
}

# Result storage
selected_features_rfe_by_cluster = {}
excluded_features_rfe_by_cluster = {}

for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")
    
    cluster_train_df = train_dfs_by_cluster[cluster_id]
    target_num_features = target_num_features_by_cluster.get(cluster_id, 5)  # fallback to 5

    # Start with features that exist in the DataFrame
    features = [f for f in numerical_columns if f in cluster_train_df.columns]
    excluded_features_rfe = []

    while len(features) > target_num_features:
        assembler = VectorAssembler(inputCols=features, outputCol="features_temp")
        assembled_df = assembler.transform(cluster_train_df).select("features_temp", "PRIMARY_DELAY_CAUSE_index")

        rf = RandomForestRegressor(featuresCol="features_temp", labelCol="PRIMARY_DELAY_CAUSE_index", seed=42)
        model = rf.fit(assembled_df)

        importances = model.featureImportances.toArray()
        feature_importance_dict = dict(zip(features, importances))

        # Remove least important feature
        least_important = sorted(feature_importance_dict.items(), key=lambda x: x[1])[0][0]
        print(f"Removed: {least_important} (importance: {feature_importance_dict[least_important]:.6f})")

        features.remove(least_important)
        excluded_features_rfe.append(least_important)

    selected_features_rfe_by_cluster[cluster_id] = features
    excluded_features_rfe_by_cluster[cluster_id] = excluded_features_rfe

    print(f"Cluster {cluster_id} - Selected features: {features}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.11.3 DT Selection

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
    features = [f for f in numerical_columns if f in cluster_train_df.columns]

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

# MAGIC %md
# MAGIC ## 5.11.4 Lasso Selection

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
    features = [f for f in numerical_columns if f in cluster_train_df.columns]

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

# MAGIC %md
# MAGIC ## 5.11.5 Majority Voting

# COMMAND ----------

from collections import Counter
from itertools import chain

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
        selected_features_spearman_by_cluster.get(cluster_id, []),
        selected_features_rfe_by_cluster.get(cluster_id, []),
        selected_features_dt_by_cluster.get(cluster_id, []),
        selected_features_lasso_by_cluster.get(cluster_id, [])
    ]

    # Flatten and count occurrences
    all_selected = list(chain.from_iterable(feature_lists))
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

# MAGIC %md
# MAGIC # 5.12 Classification

# COMMAND ----------

# MAGIC %md
# MAGIC Excluded models and reasons:
# MAGIC - GBTClassifier: Only supports binary classification in PySpark.
# MAGIC - NaiveBayes: Requires non-negative features and assumes feature independence; not ideal for general tabular data.
# MAGIC - XGBoost: Not natively available in PySpark; requires xgboost4j-spark or external integration.
# MAGIC - AdaBoost: Not implemented in PySpark MLlib; available only in scikit-learn.
# MAGIC - MultilayerPerceptron: Requires manual layer tuning, doesn't support weightCol (class weighting).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.12.1 Train, Predict, Evaluate

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, IndexToString
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import Row

# Define models with default parameters
default_models = {
    "LogisticRegression": LogisticRegression(featuresCol="features", labelCol=target_column, weightCol="weight", maxIter=100),
    "DecisionTree": DecisionTreeClassifier(featuresCol="features", labelCol=target_column, weightCol="weight", seed=42),
    "RandomForest": RandomForestClassifier(featuresCol="features", labelCol=target_column, weightCol="weight", seed=42),
}

# Evaluators
evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction", metricName="f1")
evaluator_acc = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction", metricName="accuracy")

# Containers for results and predictions
results = []
# key=(cluster_id, model_name)
train_decoded_predictions = {}  
val_decoded_predictions = {}
models_by_cluster_and_name = {}

for cluster_id in cluster_ids:
    print(f"\n--- Processing Cluster {cluster_id} ---")
    
    train_df = train_dfs_by_cluster[cluster_id]
    val_df = val_dfs_by_cluster[cluster_id]
    features = majority_voted_features_by_cluster[cluster_id]
    
    # Assemble features
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    train_prepared = assembler.transform(train_df)
    val_prepared = assembler.transform(val_df)
    
    for model_name, model in default_models.items():
        print(f"Training {model_name}...")
        
        # Fit model
        trained_model = model.fit(train_prepared)
        
        # Predict on train and val
        train_pred = trained_model.transform(train_prepared)
        val_pred = trained_model.transform(val_prepared)
        
        # Decode predictions
        train_decoded = decode_predictions(train_pred)
        val_decoded = decode_predictions(val_pred)
        
        # Save decoded predictions for confusion matrix etc.
        train_decoded_predictions[(cluster_id, model_name)] = train_decoded
        val_decoded_predictions[(cluster_id, model_name)] = val_decoded
        models_by_cluster_and_name[(cluster_id, model_name)] = trained_model
        
        # Evaluate metrics
        train_macro_f1 = compute_macro_f1(train_decoded)
        train_weighted_f1 = evaluator_f1.evaluate(train_pred)
        train_acc = evaluator_acc.evaluate(train_pred)

        val_macro_f1 = compute_macro_f1(val_decoded)
        val_weighted_f1 = evaluator_f1.evaluate(val_pred)
        val_acc = evaluator_acc.evaluate(val_pred)
        
        # Store results
        results.append(Row(
            cluster_id=cluster_id,
            model=model_name,
            train_macro_f1=train_macro_f1,
            val_macro_f1=val_macro_f1,
            overfitting_f1_macro=train_macro_f1 - val_macro_f1,
            train_weighted_f1=train_weighted_f1,
            val_weighted_f1=val_weighted_f1,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
        ))

# COMMAND ----------

from pyspark.sql.functions import format_number

# Create the DataFrame
default_results_df = spark.createDataFrame(results)

# Format numeric columns to 4 decimals for display
formatted_df = default_results_df.select(
    "cluster_id", "model",
    format_number("train_macro_f1", 4).alias("train_macro_f1"),
    format_number("val_macro_f1", 4).alias("val_macro_f1"),
    format_number("overfitting_f1_macro", 4).alias("overfitting_f1_macro"),
    format_number("train_weighted_f1", 4).alias("train_weighted_f1"),
    format_number("val_weighted_f1", 4).alias("val_weighted_f1"),
    format_number("train_accuracy", 4).alias("train_accuracy"),
    format_number("val_accuracy", 4).alias("val_accuracy"),
)

# Display overall results sorted by validation macro F1
print("--- Overall Sorted Results ---")
formatted_df.orderBy("val_macro_f1", ascending=False).display()

# Display results for each cluster separately
for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")
    formatted_df.filter(f"cluster_id = {cluster_id}") \
        .orderBy("val_macro_f1", ascending=False) \
        .display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5.12.2 Confusion Matrix

# COMMAND ----------

def plot_confusion_matrices_for_model(model_name):
    for cluster_id in cluster_ids:
        print(f"\nConfusion Matrix for Cluster {cluster_id}, Model {model_name}")

        val_decoded = val_decoded_predictions.get((cluster_id, model_name))
        
        confusion_df = val_decoded.groupBy("true_label", "predicted_label").count()
        confusion_pd = confusion_df.toPandas().pivot(
            index="true_label",
            columns="predicted_label",
            values="count"
        ).fillna(0)
        
        plt.figure(figsize=(8,6))
        sns.heatmap(confusion_pd, annot=True, fmt=".0f", cmap="Blues", cbar=True)
        plt.title(f"Confusion Matrix (Validation) - Cluster {cluster_id} - {model_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

# EPlot confusion matrices for all clusters using RandomForest
plot_confusion_matrices_for_model("RandomForest")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.13 Hyperparameter Tuning

# COMMAND ----------

# Define models with param spaces for random search
model_param_spaces = {
    "LogisticRegression": {
        "model_class": LogisticRegression,
        "param_distributions": {
            "maxIter": [50, 100, 150],
            "regParam": [0.0, 0.01, 0.1, 0.5],
            "elasticNetParam": [0.0, 0.5, 1.0]
        }
    },
    "DecisionTree": {
        "model_class": DecisionTreeClassifier,
        "param_distributions": {
            "maxDepth": [3, 5, 10, 20],
            "maxBins": [32, 64, 128],
            "minInstancesPerNode": [1, 2, 5]
        }
    },
    "RandomForest": {
        "model_class": RandomForestClassifier,
        "param_distributions": {
            "numTrees": [20, 50, 100],
            "maxDepth": [5, 10, 20],
            "maxBins": [32, 64, 128]
        }
    }
}

# COMMAND ----------

import random
import mlflow
from pyspark.ml.feature import VectorAssembler, IndexToString
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Evaluators
evaluator_f1 = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction", metricName="f1")
evaluator_acc = MulticlassClassificationEvaluator(labelCol=target_column, predictionCol="prediction", metricName="accuracy")

# Number of random search trials per model per cluster
n_trials = 60

results = []

for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")
    train_df = train_dfs_by_cluster[cluster_id]
    val_df = val_dfs_by_cluster[cluster_id]
    features = majority_voted_features_by_cluster[cluster_id]

    # Prepare data (feature vector)
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    train_prepared = assembler.transform(train_df)
    val_prepared = assembler.transform(val_df)

    for model_name, model_info in model_param_spaces.items():
        ModelClass = model_info["model_class"]
        param_dist = model_info["param_distributions"]

        for trial in range(n_trials):
            # Sample params randomly
            params = {k: random.choice(v) for k, v in param_dist.items()}

            # Create model with sampled params
            model = ModelClass(featuresCol="features", labelCol=target_column, weightCol="weight", seed=42, **params)

            with mlflow.start_run(run_name=f"Cluster{cluster_id}_{model_name}_trial{trial+1}"):
                print(f"{model_name} trial {trial+1}...")
            
                # Train
                trained_model = model.fit(train_prepared)

                # Predict on train and validation
                train_pred = trained_model.transform(train_prepared)
                val_pred = trained_model.transform(val_prepared)

                # Evaluate metrics
                train_macro_f1 = compute_macro_f1(decode_predictions(train_pred))
                val_macro_f1 = compute_macro_f1(decode_predictions(val_pred))

                train_weighted_f1 = evaluator_f1.evaluate(train_pred)
                train_accuracy = evaluator_acc.evaluate(train_pred)

                val_weighted_f1 = evaluator_f1.evaluate(val_pred)
                val_accuracy = evaluator_acc.evaluate(val_pred)                

                # Log metrics to MLflow
                mlflow.log_params(params)
                mlflow.log_metrics({
                    "train_macro_f1": train_macro_f1,
                    "val_macro_f1": val_macro_f1,
                    "overfitting_macro_f1": train_macro_f1 - val_macro_f1
                    "train_weighted_f1": train_weighted_f1,
                    "val_weighted_f1": val_weighted_f1,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                })
                mlflow.set_tag("cluster_id", cluster_id)
                mlflow.set_tag("model", model_name)
                mlflow.set_tag("trial", trial + 1)

                # Log model artifact
                mlflow.spark.log_model(trained_model, artifact_path="model")

                # Save for summary
                results.append({
                    "cluster_id": cluster_id,
                    "model": model_name,
                    **params,
                    "train_macro_f1": train_macro_f1,
                    "val_macro_f1": val_macro_f1,
                    "overfitting_macro_f1": train_macro_f1 - val_macro_f1
                    "train_weighted_f1": train_weighted_f1,
                    "val_weighted_f1": val_weighted_f1,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                    "trial": trial + 1
                })

# COMMAND ----------

from pyspark.sql.functions import format_number

# Create the DataFrame
hp_tuning_results_df = spark.createDataFrame(results)

# Format numeric columns to 4 decimals for display
formatted_df = hp_tuning_results_df.select(
    "cluster_id", "model",
    format_number("train_macro_f1", 4).alias("train_macro_f1"),
    format_number("val_macro_f1", 4).alias("val_macro_f1"),
    format_number("overfitting_f1_macro", 4).alias("overfitting_f1_macro"),
    format_number("train_weighted_f1", 4).alias("train_weighted_f1"),
    format_number("val_weighted_f1", 4).alias("val_weighted_f1"),
    format_number("train_accuracy", 4).alias("train_accuracy"),
    format_number("val_accuracy", 4).alias("val_accuracy"),
)

# Display overall results sorted by validation macro F1
print("--- Overall Sorted Results ---")
formatted_df.orderBy("val_macro_f1", ascending=False).display()

# Display results for each cluster separately
for cluster_id in cluster_ids:
    print(f"\n--- Cluster {cluster_id} ---")
    formatted_df.filter(f"cluster_id = {cluster_id}") \
        .orderBy("val_macro_f1", ascending=False) \
        .display()

# COMMAND ----------

# MAGIC %md
# MAGIC # 5.14 Export Dataframe

# COMMAND ----------

# clustered_df.write.mode("overwrite") \
#     .option("overwriteSchema", "true") \
#     .format("delta") \
#     .save("/dbfs/FileStore/tables/clustered_train_df")