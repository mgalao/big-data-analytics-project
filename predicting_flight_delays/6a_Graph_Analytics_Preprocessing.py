# Databricks notebook source
# MAGIC %md
# MAGIC # 6a. Graph Analytics - Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC # 6a.1 Import Libraries

# COMMAND ----------

from pyspark.sql.functions import lit, col, desc, abs, isnan, to_date, rand, length, count, when, hour, dayofweek, round, explode, lower, udf, mean, avg, stddev, min, max, coalesce, concat_ws, row_number, monotonically_increasing_id, floor, round as spark_round

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
spark = SparkSession.builder.appName("Graph_Analytics_Preprocessing").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC # 6a.2 Import Df

# COMMAND ----------

# Import df_eda
df = spark.read.format("delta").load("/dbfs/FileStore/tables/df_eda")

# Display result
df.limit(10).display()

# COMMAND ----------

print(f"Total rows: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 6a.3 Preprocessing From Notebook 2

# COMMAND ----------

# MAGIC %md
# MAGIC We applied all the preprocessing from the notebook "2_Preprocessing", excluding:
# MAGIC - Train-Val-Test Split
# MAGIC - Encoding

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6a.3.1 Sampling

# COMMAND ----------

df_sampled = df.sample(withReplacement=False, fraction=0.05, seed=42)

# COMMAND ----------

print(f"Total rows of the sampled dataset: {df_sampled.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6a.3.2 Treatment of numerical airport codes

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a.3.2.1 Map numeric airport codes to IATA codes

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
# MAGIC ### 6a.3.2.2 Function to replace numerical airport codes

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

cleaned_sampled_df = replace_numeric_airports(df_sampled, airport_full)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6a.3.3 Missing values

# COMMAND ----------

# MAGIC %md
# MAGIC As seen in 1_EDA, there's missing values in cancelation reason. Since the feature is not relevant for analysis, we will drop it.

# COMMAND ----------

cleaned_sampled_df = cleaned_sampled_df.drop("CANCELLATION_REASON")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6a.3.4 Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a.3.4.1 Functions

# COMMAND ----------

# MAGIC %md
# MAGIC #### Seasonality from Month
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
# MAGIC #### Departure Period from Scheduled Time
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
# MAGIC #### IS_WEEKEND
# MAGIC

# COMMAND ----------

def add_is_weekend_column(df):
    return df.withColumn("IS_WEEKEND", (col("DAY_OF_WEEK") >= 6).cast("int"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### DELAYED_DEPARTURE_FLAG

# COMMAND ----------

# MAGIC %md
# MAGIC A flight is officially considered “delayed” if its departure is more than 15 minutes late:
# MAGIC https://www.oag.com/airline-on-time-performance-defining-late#fifteenmins

# COMMAND ----------

def add_delayed_departure_flag_column(df):
    return df.withColumn("DELAYED_DEPARTURE_FLAG", (col("DEPARTURE_DELAY") > 15).cast("int"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### ROUTE

# COMMAND ----------

def add_route_column(df):
    return df.withColumn("ROUTE", concat_ws("_", col("ORIGIN_AIRPORT_CLEAN"), col("DESTINATION_AIRPORT_CLEAN")))

# COMMAND ----------

# MAGIC %md
# MAGIC #### TOTAL_KNOWN_DELAY

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
# MAGIC #### HHMM columns to minutes from midnight

# COMMAND ----------

def hhmm_to_minutes(colname):
    return (floor(col(colname) / 100) * 60 + (col(colname) % 100)).alias(colname + "_min")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a.3.4.2 Applying functions

# COMMAND ----------

cleaned_sampled_df = add_season_column(cleaned_sampled_df)
cleaned_sampled_df = add_scheduled_dep_period_column(cleaned_sampled_df)
cleaned_sampled_df = add_is_weekend_column(cleaned_sampled_df)
cleaned_sampled_df = add_delayed_departure_flag_column(cleaned_sampled_df)
cleaned_sampled_df = add_route_column(cleaned_sampled_df)
cleaned_sampled_df = add_total_known_delay(cleaned_sampled_df)

# COMMAND ----------

cleaned_sampled_df = cleaned_sampled_df.withColumn("SCHEDULED_DEPARTURE_min", hhmm_to_minutes("SCHEDULED_DEPARTURE"))
cleaned_sampled_df = cleaned_sampled_df.withColumn("SCHEDULED_ARRIVAL_min", hhmm_to_minutes("SCHEDULED_ARRIVAL"))
cleaned_sampled_df = cleaned_sampled_df.withColumn("DEPARTURE_TIME_min", hhmm_to_minutes("DEPARTURE_TIME"))
cleaned_sampled_df = cleaned_sampled_df.withColumn("WHEELS_OFF_min", hhmm_to_minutes("WHEELS_OFF"))
cleaned_sampled_df = cleaned_sampled_df.withColumn("ARRIVAL_TIME_min", hhmm_to_minutes("ARRIVAL_TIME"))
cleaned_sampled_df = cleaned_sampled_df.withColumn("DEPARTURE_TIME_min", hhmm_to_minutes("DEPARTURE_TIME"))

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

# MAGIC %md
# MAGIC ## 6a.3.5 Outliers

# COMMAND ----------

# MAGIC %md
# MAGIC As observed in the boxplot, DISTANCE contains outliers. To reduce their impact on the model, we cap the values at the 97.5th percentile.
# MAGIC

# COMMAND ----------

def cap_at_percentile(df, column, percentile):
    threshold = df.approxQuantile(column, [percentile], 0.01)[0]
    return df.withColumn(column, when(col(column) > threshold, threshold).otherwise(col(column)))


# COMMAND ----------

cleaned_sampled_df = cap_at_percentile(cleaned_sampled_df, "DISTANCE", 0.975)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6a.3.6 Coherence Checking

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a.3.6.1 Duplicated rows (after mapping airports)

# COMMAND ----------

# MAGIC %md
# MAGIC We noticed airport codes repeated so we have to treat that

# COMMAND ----------

origin_train_mapping = (
    cleaned_sampled_df.groupBy("ORIGIN_AIRPORT")
    .agg(min("ORIGIN_AIRPORT_CLEAN").alias("PREFERRED_ORIGIN_AIRPORT_CLEAN"))
)

destination_train_mapping = (
    cleaned_sampled_df.groupBy("DESTINATION_AIRPORT")
    .agg(min("DESTINATION_AIRPORT_CLEAN").alias("PREFERRED_DESTINATION_AIRPORT_CLEAN"))
)

# Join and overwrite ORIGIN_AIRPORT_CLEAN
cleaned_sampled_df = cleaned_sampled_df.join(
    origin_train_mapping, on="ORIGIN_AIRPORT", how="left"
)

# Join and overwrite DESTINATION_AIRPORT_CLEAN
cleaned_sampled_df = cleaned_sampled_df.join(
    destination_train_mapping, on="DESTINATION_AIRPORT", how="left"
)

# COMMAND ----------

cleaned_sampled_df = cleaned_sampled_df.filter(
    (col("ORIGIN_AIRPORT_CLEAN") == col("PREFERRED_ORIGIN_AIRPORT_CLEAN")) &
    (col("DESTINATION_AIRPORT_CLEAN") == col("PREFERRED_DESTINATION_AIRPORT_CLEAN"))
)

cleaned_sampled_df = cleaned_sampled_df.drop(
    "PREFERRED_ORIGIN_AIRPORT_CLEAN",
    "PREFERRED_DESTINATION_AIRPORT_CLEAN"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a.3.6.2 Other coherence checkings

# COMMAND ----------

#create a new feature, a flag, that indicates if the flight happened during midnight (1) or didn't (0), which would explain an arrival time min smaller than the departure time min, since the conversion counts the minutes since midnight
cleaned_sampled_df = cleaned_sampled_df.withColumn(
    'TOTAL_FLIGHT_MIDNIGHT_min',
    col('DEPARTURE_TIME_min') + col('AIR_TIME')
)

#function to flag flights that cross midnight
def add_crosses_midnight_flag_column(df):
    return df.withColumn('CROSSES_MIDNIGHT_FLAG', (col('TOTAL_FLIGHT_MIDNIGHT_min') >= 1440).cast("int"))

cleaned_sampled_df = add_crosses_midnight_flag_column(cleaned_sampled_df)

# COMMAND ----------

#remove rows in which departure delay is negative, the flights departure before their scheduled time
cleaned_sampled_df = cleaned_sampled_df.filter(cleaned_sampled_df.DEPARTURE_DELAY >= 0)

# COMMAND ----------

# remove rows in which the plane left before the scheduled time without being those flights that cross midnight
cleaned_sampled_df = cleaned_sampled_df.filter(
    cleaned_sampled_df.DEPARTURE_TIME_min >= cleaned_sampled_df.SCHEDULED_DEPARTURE_min
)

# COMMAND ----------

# remove rows where other time constraints are not satisfied
cleaned_sampled_df = cleaned_sampled_df.filter(
    ~(
        (col('CROSSES_MIDNIGHT_FLAG')!=1) &                               
        (col("WHEELS_OFF_min") < col("DEPARTURE_TIME_min")) | 
        (col("WHEELS_ON") < col("WHEELS_OFF")) |
        (col("ARRIVAL_TIME") < col("WHEELS_ON")) |
        (col("SCHEDULED_ARRIVAL_min") < col("SCHEDULED_DEPARTURE_min"))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6a.3.7 Drop and Rename Features

# COMMAND ----------

cleaned_sampled_df.printSchema()

# COMMAND ----------

cleaned_sampled_df.limit(10).display()

# COMMAND ----------

# Define columns to drop
cols_to_drop = [
    # 'YEAR',
    # 'MONTH',
    # 'DAY',
    'ORIGIN_AIRPORT', # wrong one
    'DESTINATION_AIRPORT', # wrong one
    "SCHEDULED_DEPARTURE", # HHMM
    "SCHEDULED_ARRIVAL", # HHMM
    "DEPARTURE_TIME", # HHMM
    "WHEELS_OFF", # HHMM
    "DELAYED_DEPARTURE_FLAG", # for graph analytics, diverted and cancelled flights will be filtered out
    "TOTAL_KNOWN_DELAY" # equal to arrival_delay
]

# Drop from DataFrames
cleaned_sampled_df = cleaned_sampled_df.drop(*cols_to_drop)

# COMMAND ----------

# Rename columns for clarity
cleaned_sampled_df = cleaned_sampled_df.withColumnRenamed("ORIGIN_AIRPORT_CLEAN", "ORIGIN_AIRPORT")
cleaned_sampled_df = cleaned_sampled_df.withColumnRenamed("DESTINATION_AIRPORT_CLEAN", "DESTINATION_AIRPORT")

# COMMAND ----------

# MAGIC %md
# MAGIC # 6a.4 Export Dataframe

# COMMAND ----------

cleaned_sampled_df.write.mode("overwrite") \
    .option("overwriteSchema", "true") \
    .format("delta") \
    .save("/dbfs/FileStore/tables/df_graph")

# COMMAND ----------

cleaned_sampled_df.limit(10).display()