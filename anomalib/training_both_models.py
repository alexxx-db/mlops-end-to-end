# Databricks notebook source
# MAGIC %md
# MAGIC ## Installing Anomalib

# COMMAND ----------

# %pip install anomalib

# COMMAND ----------

# MAGIC %md
# MAGIC  Now let's verify the working directory. This is to access the datasets and configs when the notebook is run from different platforms such as local or Google Colab.

# COMMAND ----------

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from git.repo import Repo

current_directory = Path.cwd()
if current_directory.name == "000_getting_started":
    # On the assumption that, the notebook is located in
    #   ~/anomalib/notebooks/000_getting_started/
    root_directory = current_directory.parent.parent
elif current_directory.name == "anomalib":
    # This means that the notebook is run from the main anomalib directory.
    root_directory = current_directory
else:
    # Otherwise, we'll need to clone the anomalib repo to the `current_directory`
    repo = Repo.clone_from(url="https://github.com/openvinotoolkit/anomalib.git", to_path=current_directory)
    root_directory = current_directory / "anomalib"

os.chdir(root_directory)

mvtec_dataset_root = root_directory / "datasets/MVTec"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports

# COMMAND ----------

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_lightning import Trainer
from torchvision.transforms import ToPILImage

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.models import get_model
from anomalib.pre_processing.transforms import Denormalize
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks

from anomalib.data.mvtec import MVTec, MVTecDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms


# COMMAND ----------

# MAGIC %md
# MAGIC ### DataModule
# MAGIC
# MAGIC Anomalib data modules are based on PyTorch Lightning (PL)'s `LightningDataModule` class. This class handles all the boilerplate code related to subset splitting, and creating the dataset and dataloader instances. A datamodule instance can be directly passed to a PL Trainer which is responsible for carrying out Anomalib's training/testing/inference pipelines. 
# MAGIC
# MAGIC In the current example, we will show how an Anomalib data module can be created for the MVTec Dataset, and how we can obtain training and testing dataloaders from it.
# MAGIC
# MAGIC To create a datamodule, we simply pass the path to the root folder of the dataset on the file system, together with some basic parameters related to pre-processing and image loading:

# COMMAND ----------

mvtec_bottle_datamodule = MVTec(
    root=mvtec_dataset_root,
    category="bottle",
    image_size=256,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
    task="segmentation",
    normalization=InputNormalizationMethod.NONE,  # don't apply normalization, as we want to visualize the images
)

# COMMAND ----------

mvtec_cable_datamodule = MVTec(
    root=mvtec_dataset_root,
    category="cable",
    image_size=256,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=8,
    task="segmentation",
    normalization=InputNormalizationMethod.NONE,  # don't apply normalization, as we want to visualize the images
)

# COMMAND ----------

# MAGIC %md
# MAGIC For the illustrative purposes of the current example, we need to manually call the `prepare_data` and `setup` methods. Normally it is not necessary to call these methods explicitly, as the PL Trainer would call these automatically under the hood.
# MAGIC
# MAGIC `prepare_data` checks if the dataset files can be found at the specified file system location. If not, it will download the dataset and place it in the folder.
# MAGIC
# MAGIC `setup` applies the subset splitting and prepares the PyTorch dataset objects for each of the train/val/test subsets.

# COMMAND ----------

mvtec_bottle_datamodule.prepare_data()
mvtec_bottle_datamodule.setup()

mvtec_cable_datamodule.prepare_data()
mvtec_cable_datamodule.setup()

# COMMAND ----------

# MAGIC %md
# MAGIC After the datamodule has been set up, we can use it to obtain the dataloaders of the different subsets.

# COMMAND ----------

# Train images
i, data_bottle = next(enumerate(mvtec_bottle_datamodule.train_dataloader()))
print(data_bottle.keys(), data_bottle["image"].shape)

# COMMAND ----------

# Test images
i, data_bottle = next(enumerate(mvtec_bottle_datamodule.test_dataloader()))
print(data_bottle.keys(), data_bottle["image"].shape, data_bottle["mask"].shape)

# COMMAND ----------

# Train images
i, data_cable = next(enumerate(mvtec_cable_datamodule.train_dataloader()))
print(data_cable.keys(), data_cable["image"].shape)

# COMMAND ----------

# Test images
i, data_cable = next(enumerate(mvtec_cable_datamodule.test_dataloader()))
print(data_cable.keys(), data_cable["image"].shape, data_cable["mask"].shape)

# COMMAND ----------

# MAGIC %md
# MAGIC As can be seen above, creating the dataloaders are pretty straghtforward, which could be directly used for training/testing/inference. We could visualize samples from the dataloaders as well.

# COMMAND ----------

img = ToPILImage()(data_bottle["image"][0].clone())
msk = ToPILImage()(data_bottle["mask"][0]).convert("RGB")

Image.fromarray(np.hstack((np.array(img), np.array(msk))))

# COMMAND ----------

img = ToPILImage()(data_cable["image"][0].clone())
msk = ToPILImage()(data_cable["mask"][0]).convert("RGB")

Image.fromarray(np.hstack((np.array(img), np.array(msk))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Model and Callbacks
# MAGIC
# MAGIC Now, the config file is updated as we want. We can now start model training with it. Here we will be using `datamodule`, `model` and `callbacks` to train the model. Callbacks are self-contained objects, which contains non-essential logic. This way we could inject as many callbacks as possible such as ModelLoading, Timer, Metrics, Normalization and Visualization
# MAGIC
# MAGIC In addition to the training, we would like to perform inference using OpenVINO. Therefore we will set the export configuration to openvino so that anomalib would export the trained model to the openvino format.

# COMMAND ----------

from anomalib.models import Padim

model = Padim(
    input_size=(256, 256),
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
)

from anomalib.data.task_type import TaskType
from pytorch_lightning.callbacks import ModelCheckpoint
from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode

callbacks = [
    MetricsConfigurationCallback(
        task=TaskType.SEGMENTATION,
        image_metrics=["AUROC", "F1Score"],
        pixel_metrics=["AUROC", "F1Score"]
    ),
    ModelCheckpoint(
        mode="max",
        monitor="image_AUROC",
    ),
    PostProcessingConfigurationCallback(
        normalization_method=NormalizationMethod.MIN_MAX,
        threshold_method=ThresholdMethod.ADAPTIVE,
    ),
    MinMaxNormalizationCallback(),
    
    ExportCallback(
        input_size=(256, 256),
        dirpath=str(Path.cwd()),
        filename="testmodel",
        export_mode=ExportMode.ONNX
    )
    
]


# COMMAND ----------

Path.cwd()

# COMMAND ----------

import mlflow
#from pytorch_lightning.loggers import MLFlowLogger

username = spark.sql("SELECT current_user()").first()['current_user()']

# When using databricks repos, it is not possible to write into working directories
# specifying a dbfs default dir helps to avoid this
experiment_path = f'/Users/{username}/Amgen/pytorch-lightning-on-databricks'

# We manually create the experiment so that we know the id and can send that to the worker nodes when we scale
experiment = mlflow.set_experiment(experiment_path)

#Autologging is performed when you call the fit method of pytorch_lightning.Trainer().

# Initialize a trainer
trainer = Trainer(
    callbacks=callbacks,
    accelerator="auto",
    auto_scale_batch_size=False,
    check_val_every_n_epoch=1,
    devices=1,
    gpus=None,
    max_epochs=1,
    num_sanity_val_steps=0,
    val_check_interval=1.0,
)

mlflow.pytorch.autolog()

# End any existing runs
mlflow.end_run()

# Train the model
with mlflow.start_run() as run:
  trainer.fit(model=model, datamodule=mvtec_bottle_datamodule)



# COMMAND ----------

# fetch the auto logged parameters and metrics
from mlflow import MlflowClient

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))

print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

trainer.checkpoint_callback.best_model_path

# COMMAND ----------

# load best model from checkpoint before evaluating
load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)

# '/Workspace/Repos/michelle.liu@databricks.com/anomalib/results/padim/mvtec/bottle/run/weights/model-v7.ckpt' for best_model_path

trainer.callbacks.insert(0, load_model_callback)
test_results = trainer.test(model=model, datamodule=mvtec_bottle_datamodule)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Retraining

# COMMAND ----------

# Initialize a trainer
trainer = Trainer(
    callbacks=callbacks,
    accelerator="auto",
    auto_scale_batch_size=False,
    check_val_every_n_epoch=1,
    devices=1,
    gpus=None,
    max_epochs=1,
    num_sanity_val_steps=0,
    val_check_interval=1.0,
)

trainer.fit(model=model, datamodule=mvtec_cable_datamodule)

# COMMAND ----------

# retrain the model with different datamodule
with mlflow.start_run() as run:
  trainer.fit(model=model, datamodule=mvtec_cable_datamodule)

print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

# COMMAND ----------

# load best model from checkpoint before evaluating
load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)

trainer.callbacks.insert(0, load_model_callback)
test_results = trainer.test(model=model, datamodule=mvtec_cable_datamodule)

# COMMAND ----------

trainer.checkpoint_callback.best_model_path

# COMMAND ----------


