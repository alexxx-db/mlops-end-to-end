# Databricks notebook source
# MAGIC %md
# MAGIC # Data Monitoring Quickstart for Time-Series Tables
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
# MAGIC In this notebook, we'll monitor a **time-series table**.

# COMMAND ----------

# DBTITLE 1,Install data monitoring wheel
# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_data_monitoring-0.2.0-py3-none-any.whl"

# COMMAND ----------

import databricks.data_monitoring as dm
from databricks.data_monitoring import analysis

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply("user")
username_prefix = username.split("@")[0].replace(".","_")

dbutils.widgets.text("table_name",f"{username_prefix}_airbnb_bookings_ts", "Table to Monitor")
dbutils.widgets.text("monitor_db", f"{username_prefix}_ml_db", "Output Database/Schema to use (OPTIONAL)")
dbutils.widgets.text("monitor_catalog", "dm_demo", "Unity-Catalog catalog to use (Required)")

# COMMAND ----------

# Required parameters in order to run this notebook.
CATALOG = dbutils.widgets.get("monitor_catalog")
TABLE_NAME = dbutils.widgets.get("table_name")
QUICKSTART_MONITOR_DB = dbutils.widgets.get("monitor_db") # Output database/schema to store analysis/drift metrics tables in
TIMESTAMP_COL = "timestamp"

# COMMAND ----------

# MAGIC %md ## Helper method
# MAGIC
# MAGIC This function below is for simulating data with different timestamps. You are not likely to use this function with your real production data. 

# COMMAND ----------

def write_to_delta_w_timestamp(input_df, day_step=0, table_name=TABLE_NAME, ts_col_name=TIMESTAMP_COL):
  """
  Helper function to append a timestamp and write to a delta table
  """

  # Calculate timestamp
  this_ts = (datetime.now() + timedelta(day_step)).timestamp()

  # Write in append mode
  input_df.withColumn(ts_col_name, F.lit(this_ts).cast("timestamp")) \
         .write.format("delta").mode("append") \
         .option("delta.enableChangeDataFeed", "true") \
         .saveAsTable(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC * An **existing Delta table in Unity-Catalog created/owned by current_user** containing **time-series** data:
# MAGIC   * _(REQUIRED)_`timestamp` column _(TimeStamp)_ 
# MAGIC     - used for windowing/aggregation when calculating metrics
# MAGIC * _(RECOMMENDED)_ enable Delta's [Change Data Feed](https://docs.databricks.com/delta/delta-change-data-feed.html#enable-change-data-feed) property on monitored (and baseline) table(s) for better performance
# MAGIC * _(OPTIONAL)_ an existing **baseline (Delta) table** containing same data/column names as monitored table with Change-Data-Feed property enabled
# MAGIC * _(OPTIONAL)_ an existing _(dummy)_ model in MLflow's model registry (under `models:/registry_model_name`, for links to the monitoring UI and DBSQL dashboard)
# MAGIC   - Useful for visualizing Monitoring UI if the table is linked to an ML model in MLflow registry

# COMMAND ----------

# MAGIC %md
# MAGIC To enable Change Data Feed, there are a few options:
# MAGIC <br> 
# MAGIC <br>
# MAGIC 1. At creation time
# MAGIC     - SQL: `TBLPROPERTIES (delta.enableChangeDataFeed = true)`
# MAGIC     - PySpark: `.option("delta.enableChangeDataFeed", "true")`
# MAGIC 1. Ad-hoc
# MAGIC     - SQL: `ALTER TABLE myDeltaTable SET TBLPROPERTIES (delta.enableChangeDataFeed = true)`
# MAGIC 1. Set it in your notebook session:
# MAGIC     - `%sql set spark.databricks.delta.properties.defaults.enableChangeDataFeed = true;`

# COMMAND ----------

from datetime import timedelta, datetime
from pyspark.sql import functions as F, types as T

# COMMAND ----------

# MAGIC %sql
# MAGIC -- CREATE CATALOG IF NOT EXISTS $monitor_catalog; -- If user has privileges to create one

# COMMAND ----------

# DBTITLE 1,Define catalog & set default database/schema to use
# MAGIC %sql
# MAGIC USE CATALOG $monitor_catalog;
# MAGIC CREATE SCHEMA IF NOT EXISTS $monitor_db;
# MAGIC USE $monitor_db;
# MAGIC DROP TABLE IF EXISTS $table_name;

# COMMAND ----------

# MAGIC %md
# MAGIC ## User journey
# MAGIC 1. Table Creation: Read raw table and create baseline/T1/T2/T3 tables, where T represents a specific timestamp.
# MAGIC 2. Create time-series table
# MAGIC 3. Define monitor on time-series table
# MAGIC 4. Append other time windows in batches and refresh metrics
# MAGIC 5. Create baseline table, update monitor, and backfill calculation
# MAGIC 6. [Optional] Calculate custom metrics
# MAGIC 7. [Optional] Delete the monitor
# MAGIC
# MAGIC **Note:** if you already have an existing time-series table with a timestamp column, you can skip steps 1-2.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read dataset & prepare data
# MAGIC
# MAGIC Dataset used for this example: [Airbnb price listing](http://insideairbnb.com/san-francisco/)

# COMMAND ----------

# Read data and add a unique id column (not mandatory but preferred)
raw_df = spark.read.format("parquet").load("/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/")
display(raw_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Split data
# MAGIC Split data into baseline, T, T+1, T+2 tables, where T represents a specific timestamp.

# COMMAND ----------

baseline_df, ts1_df, ts2_df, ts3_df = raw_df.randomSplit(weights=[0.20, 0.50, 0.25, 0.05], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create time-series tables

# COMMAND ----------

# Day 1 
write_to_delta_w_timestamp(ts1_df, day_step=1)

# COMMAND ----------

# Day 2
write_to_delta_w_timestamp(ts2_df, day_step=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Monitor
# MAGIC Using `TimeSeries` type analysis
# MAGIC
# MAGIC **Required parameters**:
# MAGIC - `TABLE_NAME`: Name of the table to monitor.
# MAGIC - `TIMESTAMP_COL`: Name of `timestamp` column in time-series `TABLE_NAME` table
# MAGIC - `GRANULARITIES`: Monitoring analysis granularities (i.e. `["5 minutes", "30 minutes", "1 hour", "1 day", "n weeks", "1 month", "1 year"]`)
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

# Window sizes to analyze data over
GRANULARITIES = ["1 day"]       

# Expressions to slice data with
SLICING_EXPRS = ["cancellation_policy", "accommodates > 2"]   
# LINKED_ENTITIES = [f"models:/model_registry_name"]
# DATA_MONITORING_DIR = f"/Users/{username}/DataMonitoringTEST"

# Custom Metrics
CUSTOM_METRICS = None 

# COMMAND ----------

help(dm.create_or_update_monitor)

# COMMAND ----------

# DBTITLE 1,Create monitor
print(f"Creating monitor for {TABLE_NAME}")

dm_info = dm.create_or_update_monitor(
    table_name=TABLE_NAME,
    granularities=GRANULARITIES,
    analysis_type=analysis.TimeSeries(
        timestamp_col=TIMESTAMP_COL
    ),
    output_schema_name=QUICKSTART_MONITOR_DB,
    slicing_exprs=SLICING_EXPRS 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Inspect the analysis tables

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

analysis_df = spark.sql(f"SELECT * FROM {dm_info.assets.analysis_metrics_table_name}")
display(analysis_df)

# COMMAND ----------

# MAGIC %md
# MAGIC You can see that for every column, the analysis table differentiates baseline data from other time windows and generates analyses based on:
# MAGIC - `window`
# MAGIC - `slice key`
# MAGIC
# MAGIC We can also gain insight into basic summary statistics
# MAGIC - percent_distinct
# MAGIC - data_type
# MAGIC - min
# MAGIC - max
# MAGIC - etc.

# COMMAND ----------

display(analysis_df.filter("column_name='cancellation_policy'"))

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the drift table below, we are able to tell the shifts / changes between the `ts1_df` and `ts2_df`. 

# COMMAND ----------

display(spark.sql(f"SELECT column_name, * FROM {dm_info.assets.drift_metrics_table_name}"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dm_info.assets.drift_metrics_table_name}").groupby("drift_type").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Append another consecutive time step

# COMMAND ----------

# Day 3
write_to_delta_w_timestamp(ts3_df, day_step=3)

# COMMAND ----------

dm.refresh_metrics(table_name=TABLE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create baseline table & update monitor
# MAGIC Please refer to the **FAQ section** in the [User Guide](https://drive.google.com/drive/u/0/folders/1oXuP-VleXmq0fTE4YovavboAVC7L-DF5) to understand how to best define a baseline table. In a nutshell, baseline tables should have acceptable data quality standards in their individual column distributions.

# COMMAND ----------

# OPTIONAL - for computing drift against baseline
BASELINE_TABLE = f"{TABLE_NAME}_baseline"  

# COMMAND ----------

# DBTITLE 1,Write table with CDF enabled
(baseline_df
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema",True) 
 .option("delta.enableChangeDataFeed", "true")
 .saveAsTable(f"{CATALOG}.{QUICKSTART_MONITOR_DB}.{BASELINE_TABLE}")
)

# COMMAND ----------

dm_info = dm.create_or_update_monitor(table_name=TABLE_NAME,
                                      analysis_type=analysis.TimeSeries(timestamp_col=TIMESTAMP_COL),
                                      granularities=GRANULARITIES,
                                      baseline_table_name=BASELINE_TABLE,
                                      skip_analysis=False # If True: backfill calculation with baseline data
                                     )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Since this comparison of `ts1_df` is made against the baseline `baseline_df`, we can see that `drift_type = "BASELINE"`. We will see another drift type, called `"CONSECUTIVE"`, when we have multiple time-steps to compare.

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {dm_info.assets.drift_metrics_table_name}").groupby("drift_type").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Optional] 6. Refresh metrics by adding custom metrics
# MAGIC Please refer to the **Custom Metrics** section in the [User Guide](https://drive.google.com/drive/u/0/folders/1oXuP-VleXmq0fTE4YovavboAVC7L-DF5)

# COMMAND ----------

# DBTITLE 1,Define new metrics
from pyspark.sql import types as T
from databricks.data_monitoring.metrics import Metric

CUSTOM_METRICS = [
    Metric(
           metric_type="aggregate",
           metric_name="log_avg",
           input_columns=["price"],
           metric_definition="avg(log(abs(`{{column_name}}`)+1))",
           output_type=T.DoubleType()
           ),
    Metric(
           metric_type="derived",
           metric_name="exp_log",
           input_columns=["price"],
           metric_definition="exp(log_avg)",
           output_type=T.DoubleType()
        ),
    Metric(
           metric_type="drift",
           metric_name="delta_exp",
           input_columns=["price"],
           metric_definition="{{current_df}}.exp_log - {{base_df}}.exp_log",
           output_type=T.DoubleType()
        )
]

# COMMAND ----------

# DBTITLE 1,Update monitor
dm.update_monitor(table_name=TABLE_NAME,
                  updated_params={
                   "custom_metrics":CUSTOM_METRICS
                  })

# COMMAND ----------

# MAGIC %md
# MAGIC #### Refresh metrics  & inspect dashboard
# MAGIC
# MAGIC Inspect the auto-generated monitoring [DBSQL dashboard](https://docs.databricks.com/sql/user/dashboards/index.html).

# COMMAND ----------

dm.refresh_metrics(
  table_name=TABLE_NAME,
  backfill=True # To apply new custom metrics to previous time steps
)

# COMMAND ----------

display(spark.sql(f"SELECT window, log_type, count, column_name, exp_log, log_avg FROM {dm_info.assets.analysis_metrics_table_name} WHERE COLUMN_NAME IN ('price') AND slice_key is NULL"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can inspect the drift metrics in more detail. In the following query, `window_cmp` represents the window against which `window` is being compared.

# COMMAND ----------

display(spark.sql(f"SELECT window, window_cmp, column_name, delta_exp FROM {dm_info.assets.drift_metrics_table_name} WHERE COLUMN_NAME IN ('price') AND slice_key is NULL"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## [Optional] 7. Delete the monitor
# MAGIC Uncomment the following line of code to clean up the monitor (if you wish to run the quickstart on this table again).

# COMMAND ----------

# dm.delete_monitor(table_name=TABLE_NAME, purge_artifacts=True)
