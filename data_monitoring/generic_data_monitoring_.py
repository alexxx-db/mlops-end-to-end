# Databricks notebook source
# MAGIC %md
# MAGIC # Data Monitoring Quickstart for Generic Tables
# MAGIC
# MAGIC **System requirements:**
# MAGIC - We recommend [Databricks Runtime for Machine Learning 12.2 LTS](https://docs.databricks.com/release-notes/runtime/12.2ml.html) or above
# MAGIC   - If you need to run on an older runtime, please refer to the FAQ in the User Guide
# MAGIC - [Unity-Catalog enabled workspace](https://docs.databricks.com/data-governance/unity-catalog/enable-workspaces.html)
# MAGIC - Disable **Customer-Managed Key(s)** for encryption [AWS](https://docs.databricks.com/security/keys/customer-managed-keys-managed-services-aws.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/security/keys/customer-managed-key-managed-services-azure) | [GCP]()
# MAGIC
# MAGIC [Link](https://drive.google.com/drive/u/0/folders/1oXuP-VleXmq0fTE4YovavboAVC7L-DF5) to Google Drive containing:
# MAGIC - User guide on core concepts
# MAGIC - API reference for API details and guidelines 
# MAGIC
# MAGIC In this notebook, we'll monitor a **generic table** _(with no notion of time-related changes)_.

# COMMAND ----------

# DBTITLE 1,Install data monitoring wheel
# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_data_monitoring-0.1.0-py3-none-any.whl"

# COMMAND ----------

import databricks.data_monitoring as dm
from databricks.data_monitoring import analysis

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
username_prefix = username.split("@")[0].replace(".","_")

dbutils.widgets.text("table_name",f"{username_prefix}_airbnb_bookings_generic", "Table to Monitor")
dbutils.widgets.text("baseline_table_name",f"{username_prefix}_airbnb_bookings_baseline", "Baseline table (OPTIONAL)")
dbutils.widgets.text("monitor_db", f"{username_prefix}_monitor_db", "Output Database/Schema to use (OPTIONAL)")
dbutils.widgets.text("monitor_catalog", "ajmal_demos", "Unity Catalog to use (Required)")

# COMMAND ----------

# Required parameters in order to run this notebook.
CATALOG = dbutils.widgets.get("monitor_catalog")
TABLE_NAME = dbutils.widgets.get("table_name")
QUICKSTART_MONITOR_DB = dbutils.widgets.get("monitor_db") # Output database/schema to store analysis/drift metrics tables in
BASELINE_TABLE = dbutils.widgets.get("baseline_table_name")  # OPTIONAL - Baseline table name, if any, for computing drift against baseline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC * An **existing Delta table in Unity-Catalog created/owned by current_user**
# MAGIC   - We will create a catalog in the following cells 
# MAGIC * _(OPTIONAL)_ An existing baseline Delta table containing same data/column names as monitored table
# MAGIC * _(OPTIONAL)_ an existing _(dummy)_ model in MLflow's model registry (under `models:/registry_model_name`, for links to the monitoring UI and DBSQL dashboard)
# MAGIC   - Useful for visualizing Monitoring UI if the table is linked to an ML model in MLflow registry

# COMMAND ----------

from datetime import timedelta, datetime
from pyspark.sql import functions as F, types as T

# COMMAND ----------

# MAGIC %sql
# MAGIC -- If user has privileges to create catalog
# MAGIC CREATE CATALOG IF NOT EXISTS $monitor_catalog; 

# COMMAND ----------

# DBTITLE 1,Define catalog and set default database/schema to use
# MAGIC %sql
# MAGIC USE CATALOG $monitor_catalog;
# MAGIC CREATE SCHEMA IF NOT EXISTS $monitor_db;
# MAGIC USE $monitor_db;
# MAGIC DROP TABLE IF EXISTS $table_name;

# COMMAND ----------

# MAGIC %md
# MAGIC ## User Journey
# MAGIC 1. Tables creation: Read raw data and create baseline/v1/v2 slices/dataframes
# MAGIC     1. Create **baseline table** _(OPTIONAL)_
# MAGIC     2. Create table to monitor (v1)
# MAGIC     3. Create views to INSERT/MERGE INTO
# MAGIC 2. Define monitor on (v1) table
# MAGIC 3. Execute delete/insert/merge (DML) operations & calculate or refresh metrics
# MAGIC 4. [Optional] Change schema, overwrite existing table & add custom metrics
# MAGIC 5. Inspect analysis and drift metrics tables
# MAGIC 6. [Optional] Delete the monitor
# MAGIC
# MAGIC **!** if you already have a table you can skip step 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create tables
# MAGIC Dataset used for this example: [Airbnb price listing](http://insideairbnb.com/san-francisco/)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Create baseline table
# MAGIC Please refer to the **FAQ section** in the [User Guide](https://drive.google.com/drive/u/0/folders/1oXuP-VleXmq0fTE4YovavboAVC7L-DF5) to understand how to best define a baseline table. In a nutshell, baseline tables should have acceptable data quality standards in their individual column distributions.

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS $baseline_table_name;
# MAGIC CREATE TABLE $baseline_table_name AS (
# MAGIC     SELECT * FROM parquet.`/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet`
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from $baseline_table_name

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Create/Define table to monitor
# MAGIC Read subset of data where bookings are not instantly bookable

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS $table_name;
# MAGIC CREATE TABLE $table_name AS (
# MAGIC     SELECT * FROM parquet.`/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet`
# MAGIC     WHERE instant_bookable = 't'
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Create views for ad-hoc data updates
# MAGIC
# MAGIC Below, we intentionally create both `DM_VIEW1_insert` and `DM_VIEW2_merge` with subsets of original data. In the downstream workflow, we simulate data updates by applying INSERT INTO and MERGE operations into the table so that we can refresh monitoring metrics.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- view with only non-instant_bookable and non-superhost bookings records to insert into
# MAGIC CREATE OR REPLACE VIEW DM_VIEW1_insert AS (
# MAGIC     SELECT * FROM parquet.`/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet`
# MAGIC     WHERE instant_bookable='f' AND host_is_superhost='f' 
# MAGIC );
# MAGIC
# MAGIC -- view with all super_host bookings
# MAGIC CREATE OR REPLACE VIEW DM_VIEW2_merge AS (
# MAGIC     SELECT * FROM parquet.`/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet`
# MAGIC     WHERE host_is_superhost='t' 
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create monitor
# MAGIC Using `GenericTable` type analysis
# MAGIC
# MAGIC **Required parameters**:
# MAGIC - `TABLE_NAME`: Name of the table to monitor.
# MAGIC
# MAGIC **Optional parameters**:
# MAGIC - `OUTPUT_SCHEMA_NAME`: _(OPTIONAL)_ 
# MAGIC   - Name of the database/schema where to create output tables (can be in either `{schema}` or `{catalog}.{schema}` format). 
# MAGIC   - If not provided, the default database will be used.
# MAGIC - `LINKED_ENTITIES` _(OPTIONAL)_: 
# MAGIC   - List of Databricks entity names that are associated with this table. 
# MAGIC   - **Only following entities are supported:**
# MAGIC      - `["models:/registry_model_name", "models:/my_model"]` links model(s) in the MLflow Model Registry to the monitored table.
# MAGIC   - Useful to view monitoring UI in MLflow for Private Preview
# MAGIC  
# MAGIC **Monitoring parameters**:
# MAGIC - `BASELINE_TABLE_NAME` _(OPTIONAL)_: Name of table containing baseline data
# MAGIC - `SLICING_EXPRS` _(OPTIONAL)_: List of column expressions to independently slice/group data for analysis. 
# MAGIC   - i.e. `slicing_exprs=["col_1", "col_2 > 10"]`)
# MAGIC - `CUSTOM_METRICS` _(OPTIONAL)_: A list of custom metrics to compute alongside existing aggregate, derived, and drift metrics.
# MAGIC - `SKIP_ANALYSIS` _(OPTIONAL)_: Flag to run analysis at monitor creation/update invoke time.
# MAGIC - `DATA_MONITORING_DIR` _(OPTIONAL)_: absolute path to existing directory for storing artifacts under `/{table_name}`
# MAGIC   - default = `/Users/{user_name}/databricks_data_monitoring`
# MAGIC
# MAGIC
# MAGIC **Make sure to drop any column that should be excluded from a business or use-case perspective**

# COMMAND ----------

# Expressions to slice data with
SLICING_EXPRS = ["instant_bookable", "host_is_superhost", "accommodates > 2"]   
# DATA_MONITORING_DIR = f"/Users/{username}/dm_demo"

# Custom Metrics
CUSTOM_METRICS = None

# COMMAND ----------

help(dm.create_or_update_monitor)

# COMMAND ----------

# DBTITLE 1,Create Monitor
print(f"Creating monitor for {TABLE_NAME}")

dm_info = dm.create_or_update_monitor(
    table_name=TABLE_NAME, # Or it could be f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}"
    slicing_exprs=SLICING_EXPRS
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Inspect the analysis tables

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that the cell below shows that within the monitor_db, there are four other tables:
# MAGIC
# MAGIC 1. analysis_metrics
# MAGIC 2. drift_metrics
# MAGIC
# MAGIC These two tables (analysis_metrics and drift_metrics) record the outputs of analysis jobs.

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES FROM $monitor_db LIKE '$table_name*'

# COMMAND ----------

# MAGIC %md
# MAGIC First, let's look at the `analysis_metrics` table.

# COMMAND ----------

analysisDF = spark.sql(f"SELECT column_name, slice_key, slice_value, * FROM {dm_info.assets.analysis_metrics_table_name} ORDER BY column_name, slice_key").drop("granularity")
display(analysisDF)

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that for every column, the analysis table differentiates baseline data from other time data and generates analyses based on:
# MAGIC - window
# MAGIC - slice key
# MAGIC
# MAGIC We can also gain insight into basic summary statistics
# MAGIC - percent_distinct
# MAGIC - data_type
# MAGIC - min
# MAGIC - max
# MAGIC - etc.

# COMMAND ----------

display(analysisDF.filter("column_name='cancellation_policy'"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Since there's no baseline and only one calculation (at time of monitor creation), then no drift calculations at this point.

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dm_info.assets.drift_metrics_table_name}").groupby("drift_type").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Apply changes to table and refresh metrics
# MAGIC
# MAGIC ### 3.1 INSERT INTO
# MAGIC
# MAGIC Here, we insert all the records that fulfill the filter statements `instant_bookable='f' AND host_is_superhost='f'` and refresh metrics thereafter.

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO $table_name
# MAGIC     SELECT * FROM DM_VIEW1_insert

# COMMAND ----------

# DBTITLE 1,Refresh monitoring metrics
dm.refresh_metrics(table_name=TABLE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 DELETE
# MAGIC
# MAGIC We simulate data deletion where records with `host_is_superhost='t' AND accommodates > 2` are removed.

# COMMAND ----------

# MAGIC %sql
# MAGIC DELETE FROM $table_name
# MAGIC     WHERE host_is_superhost='t' AND accommodates > 2

# COMMAND ----------

dm.refresh_metrics(table_name=TABLE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 MERGE INTO
# MAGIC
# MAGIC We merge records for `host_is_superhost='t'` whenever they match the latitude and longitude columns in the original table.

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO $table_name
# MAGIC     USING DM_View2_merge
# MAGIC     ON ($table_name.latitude = DM_View2_merge.latitude AND $table_name.longitude = DM_View2_merge.longitude)
# MAGIC     WHEN MATCHED THEN
# MAGIC         UPDATE SET *
# MAGIC     WHEN NOT MATCHED THEN
# MAGIC         INSERT *

# COMMAND ----------

dm.refresh_metrics(table_name=TABLE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Optional] 4. Overwrite table add column and re-calculate metrics by adding custom metrics
# MAGIC Please refer to the **Custom Metrics** section in the [User Guide](https://drive.google.com/drive/u/0/folders/1oXuP-VleXmq0fTE4YovavboAVC7L-DF5)

# COMMAND ----------

(spark.read.parquet("/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/") 
     .withColumn("new_numerical_col", F.randn(5000)) 
     .write.format("delta").mode("overwrite").option("overwriteSchema",True) 
     .option("delta.enableChangeDataFeed", "true") 
     .saveAsTable(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{TABLE_NAME}")
)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Define new custom metrics
from pyspark.sql import types as T
from databricks.data_monitoring.metrics import Metric


CUSTOM_METRICS = [
    Metric(
           metric_type="aggregate",
           metric_name="percentile_0.9",
           input_columns=["price"],
           metric_definition="percentile(`{{column_name}}`, 0.9)",
           output_type=T.DoubleType()
           ),
    Metric(
           metric_type="aggregate",
           metric_name="percentile_0.1",
           input_columns=["price"],
           metric_definition="percentile(`{{column_name}}`, 0.1)",
           output_type=T.DoubleType()
           ),
    Metric(
           metric_type="aggregate",
           metric_name="percentile_multiple",
           input_columns=["price"],
           metric_definition="percentile_approx(`{{column_name}}`, array(0.25, 0.5, 0.75))",
           output_type=T.ArrayType(T.DoubleType())
    )
]

# COMMAND ----------

# DBTITLE 1,Update monitor with baseline and new custom metrics
dm_info = dm.create_or_update_monitor(
  table_name=TABLE_NAME,
  baseline_table_name=BASELINE_TABLE,
  custom_metrics=CUSTOM_METRICS,
  slicing_exprs=SLICING_EXPRS
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Inspect tables
# MAGIC
# MAGIC Notice that for each invoke of `refresh_metrics`, there's a new row in the analysis tables that corresponds to the statistics per:
# MAGIC - input column
# MAGIC - input window (exact time the metrics were calculated)
# MAGIC - different values of the slicing expressions
# MAGIC
# MAGIC In particular, we can inspect the statistics for the total count of input rows for each `refresh_metrics` call and in the baseline data.

# COMMAND ----------

display(spark.sql(f"SELECT DISTINCT window, log_type, count FROM {dm_info.assets.analysis_metrics_table_name} WHERE column_name=':table' AND slice_key IS NULL"))

# COMMAND ----------

# MAGIC %md
# MAGIC In the analysis metrics table below, we can view the columns' overall statistics, including percent of distinct values, median, etc. 

# COMMAND ----------

display(spark.sql(f"SELECT column_name, * FROM {dm_info.assets.analysis_metrics_table_name} WHERE COLUMN_NAME IN ('accommodates', 'cancellation_policy', 'neighbourhood_cleansed') AND slice_key is NULL").drop("granularity", "log_type", "slice_key", "slice_value", "logging_table_commit_version"))

# COMMAND ----------

# MAGIC %md
# MAGIC Pivoting to inspecting drift metrics table, we can view the drift metrics calculated between batches of data. Since now we have different batches of scored data, we can see the `drift_type == "CONSECUTIVE"` here. Note that only the latest window has `CONSECUTIVE` drift, since it can be compared to the previous window. However, both windows have `BASELINE` drift computed.

# COMMAND ----------

display(spark.sql(f"SELECT DISTINCT(drift_type) FROM {dm_info.assets.drift_metrics_table_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can inspect the drift metrics in more detail. In the following query, `window_cmp` represents the window against which `window` is being compared.

# COMMAND ----------

display(spark.sql(f"SELECT window, window_cmp, column_name, delta_exp, * FROM {dm_info.assets.drift_metrics_table_name} WHERE COLUMN_NAME IN ('price') AND slice_key is NULL"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Optional] 6. Delete the monitor
# MAGIC Uncomment the following line of code to clean up the monitor (if you wish to run the quickstart on this table again).

# COMMAND ----------

# dm.delete_monitor(table_name=TABLE_NAME, purge_artifacts=True)
