# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC ##Deploying the model for batch inferences
# MAGIC
# MAGIC <img style="float: right; margin-left: 20px" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_batch_inference.gif" />
# MAGIC
# MAGIC Now that our model is available in the Registry, we can load it to compute our inferences and save them in a table to start building dashboards.
# MAGIC
# MAGIC We will use MLFlow function to load a pyspark UDF and distribute our inference in the entire cluster. If the data is small, we can also load the model with plain python and use a pandas Dataframe.
# MAGIC
# MAGIC If you don't know how to start, Databricks can generate a batch inference notebook in just one click from the model registry !

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the MLflow Model

# COMMAND ----------

import mlflow

model_name = "Anomalib-Pytorch-Model-1"

model = mlflow.pyfunc.spark_udf(spark, model_uri="models:/Anomalib-Pytorch-Model-1/LATEST")


# COMMAND ----------

model

# COMMAND ----------


