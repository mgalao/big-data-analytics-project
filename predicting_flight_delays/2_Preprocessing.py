# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.1 Import Libraries

# COMMAND ----------

from pyspark.sql.functions import lit, col, desc, abs, isnan, to_date, rand, length, count, when, hour, dayofweek, round, explode, lower, udf, mean, stddev, min, max, coalesce, concat_ws, row_number, monotonically_increasing_id, floor, round as spark_round

from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, BucketedRandomProjectionLSH
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window
from pyspark.sql.types import NumericType
from pyspark.sql.functions import countDistinct

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.impute import KNNImputer
from functools import reduce

import re

# COMMAND ----------

# Initialize Spark session
spark = SparkSession.builder.appName("Preprocessing").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.2 Import Df

# COMMAND ----------

# Import df_eda
df = spark.read.format("delta").load("/dbfs/FileStore/tables/df_eda")

# Display result
df.limit(10).display()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

print(f"Total rows: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.3 Sampling

# COMMAND ----------

df_sampled = df.sample(withReplacement=False, fraction=0.05, seed=42)

# COMMAND ----------

print(f"Total rows of the sampled dataset: {df_sampled.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.4 Train-Val-Test Split

# COMMAND ----------

# Add a random number per row (used to split)
df_with_rand = df_sampled.withColumn("rand", rand(seed=42))

# Create train_df/val/test thresholds
train_frac = 0.7
val_frac = 0.15
test_frac = 0.15

# Assign split label per class using window
window = Window.partitionBy("MONTH").orderBy("rand")

# Count total rows per MONTH to assign quantiles
df_with_counts = df_with_rand.withColumn("row_number", row_number().over(window)) \
    .withColumn("count_in_month", count("*").over(Window.partitionBy("MONTH")))

# Assign split based on cumulative fractions
df_split = df_with_counts.withColumn(
    "split",
    when(col("row_number") <= col("count_in_month") * train_frac, "train")
     .when(col("row_number") <= col("count_in_month") * (train_frac + val_frac), "val")
     .otherwise("test")
)

# Now split the datasets
train_df = df_split.filter(col("split") == "train").drop("rand", "row_number", "count_in_month", "split")
val_df = df_split.filter(col("split") == "val").drop("rand", "row_number", "count_in_month", "split")
test_df = df_split.filter(col("split") == "test").drop("rand", "row_number", "count_in_month", "split")


# COMMAND ----------

train_df = train_df.orderBy("DAY", "MONTH", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "FLIGHT_NUMBER", "TAIL_NUMBER")
val_df = val_df.orderBy("DAY", "MONTH", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "FLIGHT_NUMBER", "TAIL_NUMBER")
test_df = test_df.orderBy("DAY", "MONTH", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "FLIGHT_NUMBER", "TAIL_NUMBER")

window = Window.orderBy("DAY", "MONTH", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "FLIGHT_NUMBER", "TAIL_NUMBER")

train_df = train_df.withColumn("index", row_number().over(window))
val_df = val_df.withColumn("index", row_number().over(window))
test_df = test_df.withColumn("index", row_number().over(window))

cols = train_df.columns
cols = ['index'] + [c for c in cols if c != 'index']
train_df = train_df.select(cols)
val_df = val_df.select(cols)
test_df = test_df.select(cols)

# COMMAND ----------

# Count total rows per split
total_train = train_df.count()
total_val = val_df.count()
total_test = test_df.count()

print(f"Total rows in train: {total_train}")
print(f"Total rows in val: {total_val}")
print(f"Total rows in test: {total_test}")

# COMMAND ----------

print(train_df.select("index").distinct().count())
print(val_df.select("index").distinct().count())
print(test_df.select("index").distinct().count())

# COMMAND ----------

# Function to compute proportions by MONTH
def show_month_proportions(df, total_rows, name):
    proportions = df.groupBy("MONTH") \
        .count() \
        .withColumn("percentage", round((col("count") / total_rows) * 100, 2)) \
        .orderBy("MONTH")
    
    print(f"\n{name} Set - Month Distribution (%):")
    proportions.show(truncate=False)

show_month_proportions(train_df, total_train, "Train")
show_month_proportions(val_df, total_val, "Validation")
show_month_proportions(test_df, total_test, "Test")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.5 Treatment of numerical airport codes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5.1 Map numeric airport codes to IATA codes

# COMMAND ----------

# MAGIC %md
# MAGIC We map the numerical airport codes in the dataset to their corresponding IATA codes. This is achieved by importing two auxiliary datasets:
# MAGIC
# MAGIC - One with the numerical airport codes and their descriptions.
# MAGIC - Another containing the IATA codes and their descriptions.
# MAGIC
# MAGIC We perform a join on the airport description field in order to link the numerical code with the corresponding IATA code.

# COMMAND ----------

# File location and type
l_airport_file_location = "/FileStore/tables/L_AIRPORT.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","


airport = (
    spark.read.format(file_type)
    .option("inferSchema", infer_schema)
    .option("header", first_row_is_header)
    .option("sep", delimiter)
    .option("quote", '"') # To handle commas correctly
    .option("escape", '"')
    .option("multiLine", "true")  # Allow fields to span multiple lines (helps with complex quoted fields)
    .option("mode", "PERMISSIVE") # Avoid failing on corrupt records
    .load(l_airport_file_location)
)


# Display result
airport.limit(10).display()

# COMMAND ----------

# File location and type
l_airport_id_file_location = "/FileStore/tables/L_AIRPORT_ID.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","


airport_id = (
    spark.read.format(file_type)
    .option("inferSchema", infer_schema)
    .option("header", first_row_is_header)
    .option("sep", delimiter)
    .option("quote", '"') # To handle commas correctly
    .option("escape", '"')
    .option("multiLine", "true")  # Allow fields to span multiple lines (helps with complex quoted fields)
    .option("mode", "PERMISSIVE") # Avoid failing on corrupt records
    .load(l_airport_id_file_location)
)


# Display result
airport_id.limit(10).display()

# COMMAND ----------

airport_id = airport_id.withColumnRenamed("Code", "airport_id")
airport = airport.withColumnRenamed("Code", "iata_code")

# Join on Description
airport_full = airport_id.join(airport, on="Description", how="inner")

# Final selection
airport_full.select("iata_code", "airport_id", "Description")

airport_full.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5.2 Function to replace numerical airport codes

# COMMAND ----------

# MAGIC %md
# MAGIC We define a function to identify and replace numerical airport codes in the ORIGIN_AIRPORT and DESTINATION_AIRPORT columns with their corresponding IATA codes using the mapping dataframe. The function returns a cleaned dataframe with two new columns: ORIGIN_AIRPORT_CLEAN and DESTINATION_AIRPORT_CLEAN.

# COMMAND ----------

def replace_numeric_airports(df, airport_mapping_df):
    """
    Replace numeric airport codes in ORIGIN_AIRPORT and DESTINATION_AIRPORT
    with IATA codes.

    Parameters:
    df (DataFrame): input dataframe
    airport_mapping_df (DataFrame): must contain columns ['airport_code', 'iata_code']

    Returns:
    DataFrame: cleaned DataFrame with ORIGIN_AIRPORT_CLEAN and DESTINATION_AIRPORT_CLEAN
    """

    # Filter numeric origin and destination airport codes
    numeric_origin_df = df.filter(col("ORIGIN_AIRPORT").rlike("^[0-9]+")) \
                                .select(col("ORIGIN_AIRPORT").cast("string").alias("airport_code"))

    numeric_dest_df = df.filter(col("DESTINATION_AIRPORT").rlike("^[0-9]+")) \
                            .select(col("DESTINATION_AIRPORT").cast("string").alias("airport_code"))

    print("Unique numeric ORIGIN_AIRPORT codes:", numeric_origin_df.select("airport_code").distinct().count())
    print("Unique numeric DESTINATION_AIRPORT codes:", numeric_dest_df.select("airport_code").distinct().count())

    # Union and drop duplicates to get unique codes from both origin and destination
    numeric_airports_df = numeric_origin_df.union(numeric_dest_df).distinct()

    print("Total unique numeric airport codes:", numeric_airports_df.count())

    # Union and drop duplicates to get unique codes from both origin and destination
    mapped_airports = numeric_airports_df.select("airport_code").distinct() \
        .join(
            airport_full.select("airport_id", "iata_code"),
            numeric_airports_df["airport_code"] == airport_full["airport_id"],
            how="left"
        ) \
        .select("airport_code", "iata_code") \
        .dropDuplicates()

    # Join with mapping for ORIGIN and DESTINATION
    joined_df = df \
        .join(mapped_airports.withColumnRenamed("airport_code", "origin_id").withColumnRenamed("iata_code", "origin_iata"),
            df["ORIGIN_AIRPORT"] == col("origin_id"),
            how="left") \
        .join(mapped_airports.withColumnRenamed("airport_code", "dest_id").withColumnRenamed("iata_code", "dest_iata"),
            df["DESTINATION_AIRPORT"] == col("dest_id"),
            how="left")

    # Replace numeric codes with IATA codes if available
    replaced_df = joined_df.withColumn(
        "ORIGIN_AIRPORT_CLEAN",
        when(col("origin_iata").isNotNull(), col("origin_iata")).otherwise(col("ORIGIN_AIRPORT"))
    ).withColumn(
        "DESTINATION_AIRPORT_CLEAN",
        when(col("dest_iata").isNotNull(), col("dest_iata")).otherwise(col("DESTINATION_AIRPORT"))
    )

    # Filter numeric origin and destination airport codes
    numeric_origin_df = replaced_df.filter(col("ORIGIN_AIRPORT_CLEAN").rlike("^[0-9]+")) \
                                .select(col("ORIGIN_AIRPORT_CLEAN").cast("string").alias("airport_code"))

    numeric_dest_df = replaced_df.filter(col("DESTINATION_AIRPORT_CLEAN").rlike("^[0-9]+")) \
                            .select(col("DESTINATION_AIRPORT_CLEAN").cast("string").alias("airport_code"))

    # Count numeric entries
    print("Numeric ORIGIN_AIRPORT_CLEAN count:", numeric_origin_df.count())
    print("Numeric DESTINATION_AIRPORT_CLEAN count:", numeric_dest_df.count())

    # Clean up
    return replaced_df.drop("origin_id", "dest_id", "origin_iata", "dest_iata")


# COMMAND ----------

cleaned_train_df = replace_numeric_airports(train_df, airport_full)

# COMMAND ----------

# Only show rows where either origin or destination was replaced
changed_df = cleaned_train_df.filter(
    (col("ORIGIN_AIRPORT") != col("ORIGIN_AIRPORT_CLEAN")) |
    (col("DESTINATION_AIRPORT") != col("DESTINATION_AIRPORT_CLEAN"))
)

# Display only the relevant columns
changed_df.select(
    "ORIGIN_AIRPORT", "ORIGIN_AIRPORT_CLEAN",
    "DESTINATION_AIRPORT", "DESTINATION_AIRPORT_CLEAN"
).distinct().limit(10).display()

# COMMAND ----------

cleaned_val_df = replace_numeric_airports(val_df, airport_full)

# COMMAND ----------

cleaned_test_df = replace_numeric_airports(test_df, airport_full)

# COMMAND ----------

filtered_df = cleaned_train_df.filter((col("FLIGHT_NUMBER") == 2065) & (col("MONTH") == 10) & (col("DAY") == 1))
display(filtered_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.6 Missing values

# COMMAND ----------

# MAGIC %md
# MAGIC As seen in 1_EDA, there's missing values in cancelation reason. Since the feature is not relevant for analysis, we will drop it.

# COMMAND ----------

cleaned_train_df = cleaned_train_df.drop("CANCELLATION_REASON")
cleaned_val_df = cleaned_val_df.drop("CANCELLATION_REASON")
cleaned_test_df = cleaned_test_df.drop("CANCELLATION_REASON")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.7 Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.7.1 Functions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Seasonality from Month
# MAGIC To capture seasonal patterns, we created a new categorical feature SEASON based on MONTH:
# MAGIC
# MAGIC | Season  | Months                     |
# MAGIC |---------|----------------------------|
# MAGIC | Winter  | December, January, February |
# MAGIC | Spring  | March, April, May           |
# MAGIC | Summer  | June, July, August          |
# MAGIC | Autumn  | September, October, November |
# MAGIC

# COMMAND ----------

def add_season_column(df):
    return df.withColumn("SEASON",
        when(col("MONTH").isin(12, 1, 2), "Winter")
        .when(col("MONTH").isin(3, 4, 5), "Spring")
        .when(col("MONTH").isin(6, 7, 8), "Summer")
        .when(col("MONTH").isin(9, 10, 11), "Autumn"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Departure Period from Scheduled Time
# MAGIC
# MAGIC We created a new feature DEPARTURE_PERIOD based on the SCHEDULED_DEPARTURE time. The mapping is as follows:
# MAGIC
# MAGIC | Period        | Time Range         |
# MAGIC |---------------|--------------------|
# MAGIC | Early Morning | 04:00 - 07:59      |
# MAGIC | Morning       | 08:00 - 11:59      |
# MAGIC | Midday        | 12:00 - 13:59      |
# MAGIC | Afternoon     | 14:00 - 17:59      |
# MAGIC | Evening       | 18:00 - 20:59      |
# MAGIC | Night         | 21:00 - 23:59      |
# MAGIC | Late Night    | 00:00 - 03:59      |
# MAGIC

# COMMAND ----------

def add_scheduled_dep_period_column(df):
    hour = floor(col("SCHEDULED_DEPARTURE") / 100)
    return df.withColumn(
        "SCHEDULED_DEPARTURE_PERIOD",
        when((hour >= 4) & (hour <= 7), "Early Morning")
        .when((hour >= 8) & (hour <= 11), "Morning")
        .when((hour >= 12) & (hour <= 13), "Midday")
        .when((hour >= 14) & (hour <= 17), "Afternoon")
        .when((hour >= 18) & (hour <= 20), "Evening")
        .when((hour >= 21) & (hour <= 23), "Night")
        .otherwise("Late Night")  # 00:00–03:59
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ### IS_WEEKEND
# MAGIC

# COMMAND ----------

def add_is_weekend_column(df):
    return df.withColumn("IS_WEEKEND", (col("DAY_OF_WEEK") >= 6).cast("int"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### DELAYED_DEPARTURE_FLAG

# COMMAND ----------

# MAGIC %md
# MAGIC A flight is officially considered “delayed” if its departure is more than 15 minutes late:
# MAGIC https://www.oag.com/airline-on-time-performance-defining-late#fifteenmins

# COMMAND ----------

def add_delayed_departure_flag_column(df):
    return df.withColumn("DELAYED_DEPARTURE_FLAG", (col("DEPARTURE_DELAY") > 15).cast("int"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROUTE

# COMMAND ----------

def add_route_column(df):
    return df.withColumn("ROUTE", concat_ws("_", col("ORIGIN_AIRPORT_CLEAN"), col("DESTINATION_AIRPORT_CLEAN")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### TOTAL_KNOWN_DELAY

# COMMAND ----------

# MAGIC %md
# MAGIC In "Choerence Checking" section, is proved to be exactly the same as "ARRIVAL_DELAY"

# COMMAND ----------

def add_total_known_delay(df):
    return df.withColumn(
        "TOTAL_KNOWN_DELAY",
        coalesce(col("AIR_SYSTEM_DELAY"), lit(0)) +
        coalesce(col("SECURITY_DELAY"), lit(0)) +
        coalesce(col("AIRLINE_DELAY"), lit(0)) +
        coalesce(col("LATE_AIRCRAFT_DELAY"), lit(0)) +
        coalesce(col("WEATHER_DELAY"), lit(0))
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.7.2 Applying functions

# COMMAND ----------

cleaned_train_df = add_season_column(cleaned_train_df)
cleaned_train_df = add_scheduled_dep_period_column(cleaned_train_df)
cleaned_train_df = add_is_weekend_column(cleaned_train_df)
cleaned_train_df = add_delayed_departure_flag_column(cleaned_train_df)
cleaned_train_df = add_route_column(cleaned_train_df)
cleaned_train_df = add_total_known_delay(cleaned_train_df)

# COMMAND ----------

cleaned_val_df = add_season_column(cleaned_val_df)
cleaned_val_df = add_scheduled_dep_period_column(cleaned_val_df)
cleaned_val_df = add_is_weekend_column(cleaned_val_df)
cleaned_val_df = add_delayed_departure_flag_column(cleaned_val_df)
cleaned_val_df = add_route_column(cleaned_val_df)
cleaned_val_df = add_total_known_delay(cleaned_val_df)

# COMMAND ----------

cleaned_test_df = add_season_column(cleaned_test_df)
cleaned_test_df = add_scheduled_dep_period_column(cleaned_test_df)
cleaned_test_df = add_is_weekend_column(cleaned_test_df)
cleaned_test_df = add_delayed_departure_flag_column(cleaned_test_df)
cleaned_test_df = add_route_column(cleaned_test_df)
cleaned_test_df = add_total_known_delay(cleaned_test_df)

# COMMAND ----------

cleaned_train_df.limit(10).display()

# COMMAND ----------

def plot_categorical_distribution(df, column_name, title=None, xlabel=None, ylabel="Count", figsize=(8, 4)):
    """
    Plots the distribution of a categorical column from a PySpark DataFrame.

    Parameters:
        df (PySpark DataFrame): The input DataFrame.
        column_name (str): The name of the categorical column to analyze.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label (default: "Count").
        figsize (tuple): Size of the plot.
    """
    dist_df = (
        df.groupBy(column_name)
        .count()
        .toPandas()
        .sort_values("count", ascending=False)
    )

    plt.figure(figsize=figsize)
    plt.bar(dist_df[column_name], dist_df["count"])
    plt.title(title if title else f"{column_name} Distribution")
    plt.xlabel(xlabel if xlabel else column_name)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# List of categorical columns to plot
categorical_columns = [
    "SEASON",
    "SCHEDULED_DEPARTURE_PERIOD",
    "IS_WEEKEND",
    "DELAYED_DEPARTURE_FLAG",
    "ROUTE"
]

# Loop through and plot each
for column in categorical_columns:
    plot_categorical_distribution(
        cleaned_train_df,
        column_name=column,
        title=f"{column} Distribution",
        xlabel=column
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### HHMM columns to minutes from midnight

# COMMAND ----------

def hhmm_to_minutes(colname):
    return (floor(col(colname) / 100) * 60 + (col(colname) % 100)).alias(colname + "_min")

# COMMAND ----------

cleaned_train_df = cleaned_train_df.withColumn("SCHEDULED_DEPARTURE_min", hhmm_to_minutes("SCHEDULED_DEPARTURE"))
cleaned_train_df = cleaned_train_df.withColumn("SCHEDULED_ARRIVAL_min", hhmm_to_minutes("SCHEDULED_ARRIVAL"))
cleaned_train_df = cleaned_train_df.withColumn("DEPARTURE_TIME_min", hhmm_to_minutes("DEPARTURE_TIME"))
cleaned_train_df = cleaned_train_df.withColumn("WHEELS_OFF_min", hhmm_to_minutes("WHEELS_OFF"))
cleaned_train_df = cleaned_train_df.withColumn("ARRIVAL_TIME_min", hhmm_to_minutes("ARRIVAL_TIME"))
cleaned_train_df = cleaned_train_df.withColumn("DEPARTURE_TIME_min", hhmm_to_minutes("DEPARTURE_TIME"))

# COMMAND ----------

cleaned_val_df = cleaned_val_df.withColumn("SCHEDULED_DEPARTURE_min", hhmm_to_minutes("SCHEDULED_DEPARTURE"))
cleaned_val_df = cleaned_val_df.withColumn("SCHEDULED_ARRIVAL_min", hhmm_to_minutes("SCHEDULED_ARRIVAL"))
cleaned_val_df = cleaned_val_df.withColumn("DEPARTURE_TIME_min", hhmm_to_minutes("DEPARTURE_TIME"))
cleaned_val_df = cleaned_val_df.withColumn("WHEELS_OFF_min", hhmm_to_minutes("WHEELS_OFF"))
cleaned_val_df = cleaned_val_df.withColumn("ARRIVAL_TIME_min", hhmm_to_minutes("ARRIVAL_TIME"))
cleaned_val_df = cleaned_val_df.withColumn("DEPARTURE_TIME_min", hhmm_to_minutes("DEPARTURE_TIME"))

# COMMAND ----------

cleaned_test_df = cleaned_test_df.withColumn("SCHEDULED_DEPARTURE_min", hhmm_to_minutes("SCHEDULED_DEPARTURE"))
cleaned_test_df = cleaned_test_df.withColumn("SCHEDULED_ARRIVAL_min", hhmm_to_minutes("SCHEDULED_ARRIVAL"))
cleaned_test_df = cleaned_test_df.withColumn("DEPARTURE_TIME_min", hhmm_to_minutes("DEPARTURE_TIME"))
cleaned_test_df = cleaned_test_df.withColumn("WHEELS_OFF_min", hhmm_to_minutes("WHEELS_OFF"))
cleaned_test_df = cleaned_test_df.withColumn("ARRIVAL_TIME_min", hhmm_to_minutes("ARRIVAL_TIME"))
cleaned_test_df = cleaned_test_df.withColumn("DEPARTURE_TIME_min", hhmm_to_minutes("DEPARTURE_TIME"))

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.8 Outliers

# COMMAND ----------

# Get all numeric column names
numeric_features = [field.name for field in df.schema.fields if isinstance(field.dataType, NumericType)]

print("Numeric features:", numeric_features)

# COMMAND ----------

plot_df = cleaned_train_df.select(numeric_features).toPandas()

plt.figure(figsize=(20, 12))
plot_df.boxplot(rot=90)
plt.title("Box Plots of Numerical Features (Outlier Detection)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Functions

# COMMAND ----------

def cap_at_percentile(df, column, percentile):
    threshold = df.approxQuantile(column, [percentile], 0.01)[0]
    return df.withColumn(column, when(col(column) > threshold, threshold).otherwise(col(column)))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Distance

# COMMAND ----------

# Mean, std, min, max
cleaned_train_df.select(
    mean("DISTANCE").alias("mean"),
    stddev("DISTANCE").alias("stddev"),
    min("DISTANCE").alias("min"),
    max("DISTANCE").alias("max")
).show()

# percentiles
percentiles = cleaned_train_df.approxQuantile("DISTANCE", [0.25, 0.5, 0.75, 0.95, 0.975, 0.99], 0.01)
q1, q2, q3, p95, p975, p99 = percentiles
iqr = q3 - q1

print(f"Q1 (25th percentile): {q1}")
print(f"Median (50th percentile): {q2}")
print(f"Q3 (75th percentile): {q3}")
print(f"IQR of DISTANCE: {iqr}")
print(f"95th percentile: {p95}")
print(f"97.5th percentile: {p975}")
print(f"99th percentile: {p99}")

# COMMAND ----------

# Total number of rows
total_count = cleaned_train_df.count()

# Count values above 95th and 97.5th percentile
above_95_count = cleaned_train_df.filter(col("DISTANCE") > p95).count()
above_975_count = cleaned_train_df.filter(col("DISTANCE") > p975).count()

# Calculate percentages
percent_above_95 = (above_95_count / total_count) * 100
percent_above_975 = (above_975_count / total_count) * 100

print(f"Percentage of values above 95th percentile: {percent_above_95:.2f}%")
print(f"Percentage of values above 97.5th percentile: {percent_above_975:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC As observed in the boxplot, DISTANCE contains outliers. To reduce their impact on the model, we cap the values at the 97.5th percentile.
# MAGIC

# COMMAND ----------

cleaned_train_df = cap_at_percentile(cleaned_train_df, "DISTANCE", 0.975)
cleaned_val_df = cap_at_percentile(cleaned_val_df, "DISTANCE", 0.975)
cleaned_test_df = cap_at_percentile(cleaned_test_df, "DISTANCE", 0.975)

# COMMAND ----------

plot_df = cleaned_train_df.select("DISTANCE").toPandas()

plt.figure(figsize=(20, 12))
plot_df.boxplot(rot=90)
plt.title("Box Plot of DISTANCE")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.9 Encoding

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.9.1 Low Cardinality (One-Hot Encoding)

# COMMAND ----------

# List of low-cardinality features
low_cardinality_cols = ["SEASON", "SCHEDULED_DEPARTURE_PERIOD"]

# Apply StringIndexer and OneHotEncoder
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid='keep')
    for col in low_cardinality_cols]

encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec")
    for col in low_cardinality_cols]

# Build and apply the pipeline
pipeline = Pipeline(stages=indexers + encoders)
encoded_model = pipeline.fit(cleaned_train_df)

train_df_encoded = encoded_model.transform(cleaned_train_df)
val_df_encoded = encoded_model.transform(cleaned_val_df)
test_df_encoded = encoded_model.transform(cleaned_test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.9.2 High Cardinality (Frequency Encoding)

# COMMAND ----------

# Apply frequency encoding
for column in high_cardinality_cols:
    freq_df = cleaned_train_df.groupBy(column).count()
    total = cleaned_train_df.count()
    
    # Add frequency column
    freq_df = freq_df.withColumn(f"{column}_freq", round(col("count") / total, 6)).drop("count")
    
    # Compute median frequency in training data
    median_freq = freq_df.approxQuantile(f"{column}_freq", [0.5], 0.001)[0]
    
    # Join frequency back to main DataFrame (in val and test, replace the freq of unseen values with the median in traiin)
    train_df_encoded = train_df_encoded.join(freq_df, on=column, how="left")
    val_df_encoded = val_df_encoded.join(freq_df, on=column, how="left").fillna({f"{column}_freq": median_freq})
    test_df_encoded = test_df_encoded.join(freq_df, on=column, how="left").fillna({f"{column}_freq": median_freq})

# COMMAND ----------

# Show final encoded columns
encoded_cols = [f"{column}_vec" for column in low_cardinality_cols] + \
               [f"{column}_freq" for column in high_cardinality_cols]

train_df_encoded.select(encoded_cols).limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Unseen Values in Val

# COMMAND ----------

# High-cardinality columns
high_cardinality_cols = ["FLIGHT_NUMBER","TAIL_NUMBER", "AIRLINE", "ORIGIN_AIRPORT_CLEAN", "DESTINATION_AIRPORT_CLEAN", "ROUTE"]

# COMMAND ----------

from pyspark.sql.functions import expr, col

freq_columns = [f"{col}_freq" for col in high_cardinality_cols]

print(f"Median of _freq columns in train:")
for freq_col in freq_columns:
    median_val = train_df_encoded.approxQuantile(freq_col, [0.5], 0.001)[0]
    print(f"{freq_col}: {median_val}")

# COMMAND ----------

from pyspark.sql.functions import col, desc, round

# total rows in val to calculate frequency
val_total = val_df_encoded.count()  

for col_name in high_cardinality_cols:
    print(f"Top 5 unseen values in val for column: {col_name}")
    
    # Distinct train values
    train_values = cleaned_train_df.select(col_name).distinct()
    
    # Count per val value
    val_counts = val_df_encoded.groupBy(col_name).count()
    
    # Unseen in train
    unseen_vals = val_counts.join(train_values, on=col_name, how='left_anti')
    
    # Add relative frequency column, sort and show top 5
    unseen_vals = unseen_vals.withColumn("frequency", round(col("count") / val_total, 6)) \
                             .orderBy(desc("count")) \
                             .limit(5)
    
    unseen_vals.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.10 Coherence Checking

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.10.1 Duplicated rows (after mapping airports)

# COMMAND ----------

# MAGIC %md
# MAGIC We noticed airport codes repeated so we have to treat that

# COMMAND ----------

duplicate_indexes_train = (
    train_df_encoded.groupBy("index")
    .agg(count("*").alias("count"))
    .filter(col("count") > 1)
)

print("Duplicate indexes in train_df:")
duplicate_indexes_train.limit(10).display()

# COMMAND ----------

duplicate_rows = train_df_encoded.join(
    duplicate_indexes_train.select("index"),
    on="index",
    how="inner")
    
columns_to_show = [
    "index", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "ORIGIN_AIRPORT_CLEAN", "DESTINATION_AIRPORT_CLEAN"
]

duplicate_rows.select(columns_to_show).limit(10).display()

# COMMAND ----------

origin_train_mapping = (
    train_df_encoded.groupBy("ORIGIN_AIRPORT")
    .agg(min("ORIGIN_AIRPORT_CLEAN").alias("PREFERRED_ORIGIN_AIRPORT_CLEAN"))
)

destination_train_mapping = (
    train_df_encoded.groupBy("DESTINATION_AIRPORT")
    .agg(min("DESTINATION_AIRPORT_CLEAN").alias("PREFERRED_DESTINATION_AIRPORT_CLEAN"))
)

# Join and overwrite ORIGIN_AIRPORT_CLEAN
train_df_encoded = train_df_encoded.join(
    origin_train_mapping, on="ORIGIN_AIRPORT", how="left"
)

# Join and overwrite DESTINATION_AIRPORT_CLEAN
train_df_encoded = train_df_encoded.join(
    destination_train_mapping, on="DESTINATION_AIRPORT", how="left"
)

# COMMAND ----------

duplicate_rows = train_df_encoded.join(
    duplicate_indexes_train.select("index"),
    on="index",
    how="inner")

columns_to_show = [
    "index", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "ORIGIN_AIRPORT_CLEAN", "DESTINATION_AIRPORT_CLEAN", 
    "PREFERRED_ORIGIN_AIRPORT_CLEAN", "PREFERRED_DESTINATION_AIRPORT_CLEAN"
]

duplicate_rows.select(columns_to_show).limit(10).display()

# COMMAND ----------

cleaned_train_df = train_df_encoded.filter(
    (col("ORIGIN_AIRPORT_CLEAN") == col("PREFERRED_ORIGIN_AIRPORT_CLEAN")) &
    (col("DESTINATION_AIRPORT_CLEAN") == col("PREFERRED_DESTINATION_AIRPORT_CLEAN"))
)

cleaned_train_df = cleaned_train_df.drop(
    "PREFERRED_ORIGIN_AIRPORT_CLEAN",
    "PREFERRED_DESTINATION_AIRPORT_CLEAN"
)

# COMMAND ----------

origin_val_mapping = (
    val_df_encoded.groupBy("ORIGIN_AIRPORT")
    .agg(min("ORIGIN_AIRPORT_CLEAN").alias("PREFERRED_ORIGIN_AIRPORT_CLEAN"))
)

destination_val_mapping = (
    val_df_encoded.groupBy("DESTINATION_AIRPORT")
    .agg(min("DESTINATION_AIRPORT_CLEAN").alias("PREFERRED_DESTINATION_AIRPORT_CLEAN"))
)

# Join and overwrite ORIGIN_AIRPORT_CLEAN
val_df_encoded = val_df_encoded.join(
    origin_val_mapping, on="ORIGIN_AIRPORT", how="left"
)

# Join and overwrite DESTINATION_AIRPORT_CLEAN
val_df_encoded = val_df_encoded.join(
    destination_val_mapping, on="DESTINATION_AIRPORT", how="left"
)

cleaned_val_df = val_df_encoded.filter(
    (col("ORIGIN_AIRPORT_CLEAN") == col("PREFERRED_ORIGIN_AIRPORT_CLEAN")) &
    (col("DESTINATION_AIRPORT_CLEAN") == col("PREFERRED_DESTINATION_AIRPORT_CLEAN"))
)

cleaned_val_df = cleaned_val_df.drop(
    "PREFERRED_ORIGIN_AIRPORT_CLEAN",
    "PREFERRED_DESTINATION_AIRPORT_CLEAN"
)

# COMMAND ----------

origin_test_mapping = (
    test_df_encoded.groupBy("ORIGIN_AIRPORT")
    .agg(min("ORIGIN_AIRPORT_CLEAN").alias("PREFERRED_ORIGIN_AIRPORT_CLEAN"))
)

destination_test_mapping = (
    test_df_encoded.groupBy("DESTINATION_AIRPORT")
    .agg(min("DESTINATION_AIRPORT_CLEAN").alias("PREFERRED_DESTINATION_AIRPORT_CLEAN"))
)

# Join and overwrite ORIGIN_AIRPORT_CLEAN
test_df_encoded = test_df_encoded.join(
    origin_test_mapping, on="ORIGIN_AIRPORT", how="left"
)

# Join and overwrite DESTINATION_AIRPORT_CLEAN
test_df_encoded = test_df_encoded.join(
    destination_test_mapping, on="DESTINATION_AIRPORT", how="left"
)

cleaned_test_df = test_df_encoded.filter(
    (col("ORIGIN_AIRPORT_CLEAN") == col("PREFERRED_ORIGIN_AIRPORT_CLEAN")) &
    (col("DESTINATION_AIRPORT_CLEAN") == col("PREFERRED_DESTINATION_AIRPORT_CLEAN"))
)

cleaned_test_df = cleaned_test_df.drop(
    "PREFERRED_ORIGIN_AIRPORT_CLEAN",
    "PREFERRED_DESTINATION_AIRPORT_CLEAN"
)

# COMMAND ----------

total_train_rows = cleaned_train_df.count()
total_val_rows = cleaned_val_df.count()
total_test_rows = cleaned_test_df.count()

print(f"train_df → total: {total_train_rows}")
print(f"val_df → total: {total_val_rows}")
print(f"test_df → total: {total_test_rows}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Other coherence checkings

# COMMAND ----------

#checking if all datasets have the same number of columns
train_cols_count = len(cleaned_train_df.columns)
val_cols_count = len(cleaned_val_df.columns)
test_cols_count = len(cleaned_test_df.columns)

if train_cols_count == val_cols_count == test_cols_count:
    print("All datasets have the same number of columns")
else:
    print("Datasets have different columns")


# COMMAND ----------

#check if the columns and datatypes match
train_cols = set(cleaned_train_df.dtypes)
val_cols = set(cleaned_val_df.dtypes)
test_cols = set(cleaned_test_df.dtypes)

if train_cols == val_cols == test_cols:
    print("All datasets have matching columns and data types.")
else:
    print("There are differences in columns or data types.")

# COMMAND ----------

#checking types
cleaned_train_df.dtypes

# COMMAND ----------

#function to check the presence of NaN values or null values
def check_missing_values(df, df_name):
    print(f"Missing values in {df_name}:")

    # Split columns by type
    numeric_types = ['int', 'double', 'bigint', 'float']
    string_cols = [c for c, t in df.dtypes if t == 'string']
    numeric_cols = [c for c, t in df.dtypes if t in numeric_types]
    vector_cols = [field.name for field in df.schema.fields if isinstance(field.dataType, VectorUDT)]

    #check Null and NaN for numeric
    numeric_missing = df.select([
        count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) for c in numeric_cols
    ])

    # check only null for string
    string_missing = df.select([
        count(when(col(c).isNull(), c)).alias(c) for c in string_cols
    ])

    # check nulls in vector columns 
    vector_missing = df.select([
        count(when(col(c).isNull(), c)).alias(c) for c in vector_cols
    ])

    # Display both
    print("Numeric columns:")
    display(numeric_missing)
    print("String columns:")
    display(string_missing)
    print("Vector columns:")
    display(vector_missing)


# COMMAND ----------

check_missing_values(cleaned_train_df, "cleaned_train_df")
check_missing_values(cleaned_val_df, "cleaned_val_df")
check_missing_values(cleaned_test_df, "cleaned_test_df")


# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: Missing values exist in the val and test datasets, in the columns: FLIGHT_NUMBER_FREQ, TAIL_NUMBER_FREQ, ORIGIN_AIRPORT_CLEAN_FREQ,DESTINATION_AIRPORT_CLEAN_FREQ, ROUTE_FREQ, columns in which the values are dependent of the context of each row so filling them with the mode of the train dataset would be innacurate. The best strategy is leave them unfilled.

# COMMAND ----------

#check the number of unique values in each column
unique_counts = cleaned_train_df.select([
    countDistinct(col).alias(col) for col in cleaned_train_df.columns
])
display(unique_counts)

# COMMAND ----------

#check if the values in each column make sense
for col_name in cleaned_train_df.columns:
    print(f"Unique values in column: {col_name}")
    cleaned_train_df.select(col_name).distinct().limit(10).display()

# COMMAND ----------

numeric_types = ['int', 'bigint', 'double']
numeric_cols = [c for c, t in cleaned_train_df.dtypes if t in numeric_types]

stats = []
for c in numeric_cols:
    zero_count = cleaned_train_df.filter(col(c) == 0).count()
    neg_count = cleaned_train_df.filter(col(c) < 0).count()
    stats.append((c, zero_count > 0, neg_count > 0, zero_count, neg_count))

result_df = spark.createDataFrame(stats, schema=["column", "has_zero", "has_negative", "zero_count", "negative_count"])
result_df.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC **Time Columns - Coherence check**

# COMMAND ----------

#create a new feature, a flag, that indicates if the flight happened during midnight (1) or didn't (0), which would explain an arrival time min smaller than the departure time min, since the conversion counts the minutes since midnight
cleaned_train_df = cleaned_train_df.withColumn(
    'TOTAL_FLIGHT_MIDNIGHT_min',
    col('DEPARTURE_TIME_min') + col('AIR_TIME')
)

cleaned_val_df = cleaned_val_df.withColumn(
    'TOTAL_FLIGHT_MIDNIGHT_min',
    col('DEPARTURE_TIME_min') + col('AIR_TIME')
)

cleaned_test_df = cleaned_test_df.withColumn(
    'TOTAL_FLIGHT_MIDNIGHT_min',
    col('DEPARTURE_TIME_min') + col('AIR_TIME')
)
#function to flag flights that cross midnight
def add_crosses_midnight_flag_column(df):
    return df.withColumn('CROSSES_MIDNIGHT_FLAG', (col('TOTAL_FLIGHT_MIDNIGHT_min') >= 1440).cast("int"))

cleaned_train_df=add_crosses_midnight_flag_column(cleaned_train_df)
cleaned_val_df=add_crosses_midnight_flag_column(cleaned_val_df)
cleaned_test_df=add_crosses_midnight_flag_column(cleaned_test_df)

cleaned_train_df[['DEPARTURE_TIME', 'AIR_TIME', 'CROSSES_MIDNIGHT_FLAG']].limit(10).display()

# COMMAND ----------

#rows in which departure delay is negative, the flights departure before their scheduled time
cleaned_train_df.filter(cleaned_train_df.DEPARTURE_DELAY < 0).limit(10).display()

#remove these rows
cleaned_train_df = cleaned_train_df.filter(cleaned_train_df.DEPARTURE_DELAY >= 0)

# COMMAND ----------

#check if there are more rows in which the plane left before the scheduled time without being those flights that cross midnight
early_departures = cleaned_train_df.filter(
    (col("DEPARTURE_TIME_min") < col("SCHEDULED_DEPARTURE_min"))  & 
    (col('CROSSES_MIDNIGHT_FLAG')!=1)
)

early_departures_without_condition = cleaned_train_df.filter(
    (col("DEPARTURE_TIME_min") < col("SCHEDULED_DEPARTURE_min"))  
)
early_departures.limit(10).display()
early_departures_without_condition.limit(10).display() #it is the same number of rows as with the condition of the flights don't cross midnight

#remove these rows
cleaned_train_df = cleaned_train_df.filter(
    cleaned_train_df.DEPARTURE_TIME_min >= cleaned_train_df.SCHEDULED_DEPARTURE_min
)

# COMMAND ----------

#check if there are no delays then the scheduled arrival is equal to the arrival time
cleaned_train_df.filter(
    (cleaned_train_df.AIR_SYSTEM_DELAY == 0) &
    (cleaned_train_df.SECURITY_DELAY == 0) &
    (cleaned_train_df.AIRLINE_DELAY == 0) &
    (cleaned_train_df.LATE_AIRCRAFT_DELAY == 0) &
    (cleaned_train_df.WEATHER_DELAY == 0) &
    ((cleaned_train_df.SCHEDULED_ARRIVAL != cleaned_train_df.ARRIVAL_TIME) | (cleaned_train_df.SCHEDULED_DEPARTURE != cleaned_train_df.DEPARTURE_TIME))
).limit(10).display()
# 0 rows, no incoherence

# COMMAND ----------

#check other time constraints that should be satisfied
  
invalid_flights_df = cleaned_train_df.filter(    
    (col('CROSSES_MIDNIGHT_FLAG')!=1) &                               
    (col("WHEELS_OFF_min") < col("DEPARTURE_TIME_min")) | 
    (col("WHEELS_ON") < col("WHEELS_OFF")) |
    (col("ARRIVAL_TIME") < col("WHEELS_ON")) |
    (col("SCHEDULED_ARRIVAL_min") < col("SCHEDULED_DEPARTURE_min")) 
)

invalid_flights_df.select("AIR_TIME", "DEPARTURE_TIME").distinct().limit(10).display()

display(invalid_flights_df.count())

#remove these rows
cleaned_train_df = cleaned_train_df.filter(
    ~(
        (col('CROSSES_MIDNIGHT_FLAG')!=1) &                               
        (col("WHEELS_OFF_min") < col("DEPARTURE_TIME_min")) | 
        (col("WHEELS_ON") < col("WHEELS_OFF")) |
        (col("ARRIVAL_TIME") < col("WHEELS_ON")) |
        (col("SCHEDULED_ARRIVAL_min") < col("SCHEDULED_DEPARTURE_min"))
    )
)


# COMMAND ----------

#check if the scheduled departure plus the delay equals the departure time
match_df = cleaned_train_df.filter(
    col("SCHEDULED_DEPARTURE_min") + col("DEPARTURE_DELAY") == col("DEPARTURE_TIME_min")
)

match_df.count()
#no incoherence

# COMMAND ----------

#check if the scheduled arrival plus the delay equals the actual arrival time
match_arrival = cleaned_train_df.filter(
    col("SCHEDULED_ARRIVAL_min") + col("ARRIVAL_DELAY") != col("ARRIVAL_TIME_min")
)

match_arrival[['SCHEDULED_ARRIVAL_min', 'ARRIVAL_DELAY', 'ARRIVAL_TIME_min', 'CROSSES_MIDNIGHT_FLAG']].limit(10).display()
        
match_arrival.count()
#the rows in which this isn't satisfied are columns where the scheduled arrival is before midnight but the arrival time is after midnight, this formula doesn't take this into account, so we will proceed to a further check, to guarentee their correctness
match_arrival = match_arrival.withColumn(
    'TOTAL_ARRIVAL_UNTIL_MIDNIGHT',
    col('SCHEDULED_ARRIVAL_min') + col('ARRIVAL_DELAY')
)

match_arrival = match_arrival.withColumn(
    'MINUTES_LEFT_AFTER_MIDNIGHT',
    col('TOTAL_ARRIVAL_UNTIL_MIDNIGHT') - 1440
)

check_remaining_rows=match_arrival.filter(
    col("MINUTES_LEFT_AFTER_MIDNIGHT") != col("ARRIVAL_TIME_min")
)

check_remaining_rows.count()
#no incoherence

# COMMAND ----------


#arrival delay must be equal to total_known_delay, check this condition for all rows
diff_count = cleaned_train_df.filter(col("ARRIVAL_DELAY") != col("TOTAL_KNOWN_DELAY")).count()
total_count = cleaned_train_df.count()

print(f"Rows with ARRIVAL_DELAY != TOTAL_KNOWN_DELAY: {diff_count}")
print(f"Total rows: {total_count}")
print(f"Percentage: {diff_count / total_count * 100:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.11 Drop Features and Feature Treatment

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
# MAGIC | ARRIVAL_TIME         | Actual arrival time — directly reveals the target.                                   |
# MAGIC | TAXI_IN              | Time spent taxiing after landing — only known after arrival.                         |
# MAGIC | WHEELS_ON            | Time the aircraft touched the runway — post-flight metric.                           |
# MAGIC | ELAPSED_TIME         | Total actual time of flight — includes arrival info.                                 |
# MAGIC | CANCELLED            | Indicates flight didn’t operate — not relevant when flight has already departed.     |
# MAGIC | DIVERTED             | Only known after the flight ends — may distort predictions if included.              |
# MAGIC | AIR_TIME             | Only known after the flight ends — reflects the actual flying duration from WHEELS_OFF to WHEELS_ON             |
# MAGIC
# MAGIC We do not exclude ARRIVAL_DELAY, TOTAL_KNOWN_DELAY, AIR_SYSTEM_DELAY, SECURITY_DELAY, AIRLINE_DELAY, LATE_AIRCRAFT_DELAY and WEATHER_DELAY because these features are going to be used later on to define target variables.

# COMMAND ----------

cleaned_train_df.columns

# COMMAND ----------

# List of columns to drop
cols_to_drop = [
    "ARRIVAL_TIME",
    "TAXI_IN",
    "WHEELS_ON",
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

# Define columns to drop
cols_to_drop = [
    # 'YEAR',
    # 'MONTH',
    # 'DAY',
    'SEASON_index',
    'SCHEDULED_DEPARTURE_PERIOD_index',
    'TAIL_NUMBER', 
    'ORIGIN_AIRPORT', # wrong one
    'DESTINATION_AIRPORT', # wrong one
    "SCHEDULED_DEPARTURE", # HHMM
    "SCHEDULED_ARRIVAL", # HHMM
    "DEPARTURE_TIME", # HHMM
    "WHEELS_OFF" # HHMM
]

# Drop from DataFrames
cleaned_train_df = cleaned_train_df.drop(*cols_to_drop)
cleaned_val_df = cleaned_val_df.drop(*cols_to_drop)
cleaned_test_df = cleaned_test_df.drop(*cols_to_drop)

cols_to_drop

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.12 Rename Columns

# COMMAND ----------

# Rename columns for clarity
cleaned_train_df = cleaned_train_df.withColumnRenamed("ORIGIN_AIRPORT_CLEAN_freq", "ORIGIN_AIRPORT_freq")
cleaned_train_df = cleaned_train_df.withColumnRenamed("DESTINATION_AIRPORT_CLEAN_freq", "DESTINATION_AIRPORT_freq")
cleaned_val_df = cleaned_val_df.withColumnRenamed("ORIGIN_AIRPORT_CLEAN_freq", "ORIGIN_AIRPORT_freq")
cleaned_val_df = cleaned_val_df.withColumnRenamed("DESTINATION_AIRPORT_CLEAN_freq", "DESTINATION_AIRPORT_freq")
cleaned_test_df = cleaned_test_df.withColumnRenamed("ORIGIN_AIRPORT_CLEAN_freq", "ORIGIN_AIRPORT_freq")
cleaned_test_df = cleaned_test_df.withColumnRenamed("DESTINATION_AIRPORT_CLEAN_freq", "DESTINATION_AIRPORT_freq")

# COMMAND ----------

cleaned_train_df = cleaned_train_df.withColumnRenamed("ORIGIN_AIRPORT_CLEAN", "ORIGIN_AIRPORT")
cleaned_train_df = cleaned_train_df.withColumnRenamed("DESTINATION_AIRPORT_CLEAN", "DESTINATION_AIRPORT")
cleaned_val_df = cleaned_val_df.withColumnRenamed("ORIGIN_AIRPORT_CLEAN", "ORIGIN_AIRPORT")
cleaned_val_df = cleaned_val_df.withColumnRenamed("DESTINATION_AIRPORT_CLEAN", "DESTINATION_AIRPORT")
cleaned_test_df = cleaned_test_df.withColumnRenamed("ORIGIN_AIRPORT_CLEAN", "ORIGIN_AIRPORT")
cleaned_test_df = cleaned_test_df.withColumnRenamed("DESTINATION_AIRPORT_CLEAN", "DESTINATION_AIRPORT")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2.13 Export Dataframe

# COMMAND ----------

# Export train DataFrame as a Delta table
cleaned_train_df.write.mode("overwrite") \
    .option("overwriteSchema", "true") \
    .format("delta") \
    .save("/dbfs/FileStore/tables/train_df")

# Export val DataFrame as a Delta tablcleaned_e
cleaned_val_df.write.mode("overwrite") \
    .option("overwriteSchema", "true") \
    .format("delta") \
    .save("/dbfs/FileStore/tables/val_df")

# Export test DataFrame as a Delta tablecleaned_
cleaned_test_df.write.mode("overwrite") \
    .option("overwriteSchema", "true") \
    .format("delta") \
    .save("/dbfs/FileStore/tables/test_df")


# COMMAND ----------

cleaned_train_df.limit(10).display()

# COMMAND ----------

cleaned_train_df.count()

# COMMAND ----------

cleaned_val_df.count()

# COMMAND ----------

cleaned_test_df.count()