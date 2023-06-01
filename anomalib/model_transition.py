# Databricks notebook source
# MAGIC %md
# MAGIC ### Register Latest Model to Registry

# COMMAND ----------

# MAGIC %md
# MAGIC #### Sending our model to the registry
# MAGIC
# MAGIC We select the last model from our experiment run and deploy it in the registry. We can easily do that using MLFlow `search_runs` API:

# COMMAND ----------

import mlflow

#Let's get our last auto ml run. This is specific to the demo, it just gets the experiment ID of the last Auto ML run.
experiment_id = "244528969818379"

best_model = mlflow.search_runs(experiment_ids=[experiment_id], max_results=1, filter_string="status = 'FINISHED'")
best_model

# COMMAND ----------

# MAGIC %md Once we have our best model, we can now deploy it in production using it's run ID

# COMMAND ----------

run_id = best_model.iloc[0]['run_id']

model_name = "Anomalib-Pytorch-Model-1"

#Deploy our autoML run in MLFlow registry
model_details = mlflow.register_model(f"runs:/{run_id}/model", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC At this point the model will be in `None` stage.  Let's update the description before moving it to `Staging`.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Update Description
# MAGIC We'll do this for the registered model overall, and the particular version.

# COMMAND ----------

from datetime import datetime
client = mlflow.tracking.MlflowClient()

#Gives more details on this specific model version
client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was retrained on new data."
)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #### Request Transition to Staging
# MAGIC
# MAGIC <img style="float: right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn_move_to_stating.gif">
# MAGIC
# MAGIC Our model is now read! Let's request a transition to Staging. 
# MAGIC
# MAGIC While this example is done using the API, we can also simply click on the Model Registry button.

# COMMAND ----------

import urllib 
import json 
import requests, json
from mlflow.utils.rest_utils import http_request

host_creds = client._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()

# Request transition to staging (API call)
def request_transition(model_name, version, stage):
  
  staging_request = {'name': model_name,
                     'version': version,
                     'stage': stage,
                     'archive_existing_versions': 'true'}
  response = mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))
  return(response)

# COMMAND ----------

request_transition(model_name = model_name, version = model_details.version, stage = "Staging")
