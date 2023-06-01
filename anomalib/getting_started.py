# Databricks notebook source
# MAGIC %md
# MAGIC <center><img src="https://raw.githubusercontent.com/openvinotoolkit/anomalib/main/docs/source/images/logos/anomalib-wide-blue.png" alt="Paris" class="center"></center>
# MAGIC
# MAGIC <center>💙 A library for benchmarking, developing and deploying deep learning anomaly detection algorithms</center>
# MAGIC
# MAGIC ______________________________________________________________________
# MAGIC
# MAGIC > NOTE:
# MAGIC > This notebook is originally created by @innat on [Kaggle](https://www.kaggle.com/code/ipythonx/mvtec-ad-anomaly-detection-with-anomalib-library/notebook).
# MAGIC
# MAGIC [Anomalib](https://github.com/openvinotoolkit/anomalib): Anomalib is a deep learning library that aims to collect state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets. Anomalib provides several ready-to-use implementations of anomaly detection algorithms described in the recent literature, as well as a set of tools that facilitate the development and implementation of custom models. The library has a strong focus on image-based anomaly detection, where the goal of the algorithm is to identify anomalous images, or anomalous pixel regions within images in a dataset.
# MAGIC
# MAGIC The library supports [`MVTec AD`](https://www.mvtec.com/company/research/datasets/mvtec-ad) (CC BY-NC-SA 4.0) and [`BeanTech`](https://paperswithcode.com/dataset/btad) (CC-BY-SA) for **benchmarking** and `folder` for custom dataset **training/inference**. In this notebook, we will explore `anomalib` training a PADIM model on the `MVTec AD` bottle dataset and evaluating the model's performance. The sections in this notebook explores the steps in `tools/train.py` more in detail. Those who would like to reproduce the results via CLI could use `python tools/train.py --model padim`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Installing Anomalib

# COMMAND ----------

# MAGIC %md
# MAGIC Installation can be done in two ways: (i) install via PyPI, or (ii) installing from sourc, both of which are shown below:

# COMMAND ----------

# MAGIC %md
# MAGIC ### I. Install via PyPI

# COMMAND ----------

# Option - I: Uncomment the next line if you want to install via pip.
%pip install anomalib
%pip install onnx
%pip install -U 'mlflow>=1.0.0'

# COMMAND ----------

# MAGIC %md
# MAGIC ### II. Install from Source
# MAGIC This option would initially download anomalib repository from github and manually install `anomalib` from source, which is shown below:

# COMMAND ----------

# Option - II: Uncomment the next three lines if you want to install from the source.
# !git clone https://github.com/openvinotoolkit/anomalib.git
# %cd anomalib
# %pip install .

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

# COMMAND ----------

root_directory

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model
# MAGIC
# MAGIC Currently, there are **13** anomaly detection models available in `anomalib` library. Namely,
# MAGIC
# MAGIC - [CFA](https://arxiv.org/abs/2206.04325)
# MAGIC - [CS-Flow](https://arxiv.org/abs/2110.02855v1)
# MAGIC - [CFlow](https://arxiv.org/pdf/2107.12571v1.pdf)
# MAGIC - [DFKDE](https://github.com/openvinotoolkit/anomalib/tree/main/anomalib/models/dfkde)
# MAGIC - [DFM](https://arxiv.org/pdf/1909.11786.pdf)
# MAGIC - [DRAEM](https://arxiv.org/abs/2108.07610)
# MAGIC - [FastFlow](https://arxiv.org/abs/2111.07677)
# MAGIC - [Ganomaly](https://arxiv.org/abs/1805.06725)
# MAGIC - [Padim](https://arxiv.org/pdf/2011.08785.pdf)
# MAGIC - [Patchcore](https://arxiv.org/pdf/2106.08265.pdf)
# MAGIC - [Reverse Distillation](https://arxiv.org/abs/2201.10703)
# MAGIC - [R-KDE](https://ieeexplore.ieee.org/document/8999287)
# MAGIC - [STFPM](https://arxiv.org/pdf/2103.04257.pdf)
# MAGIC
# MAGIC In this tutorial, we'll be using Padim. Now, let's get their config paths from the respected folders.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC In this demonstration, we will choose [Padim](https://arxiv.org/pdf/2011.08785.pdf) model from the above list. Let's take a quick look at its config file.

# COMMAND ----------

MODEL = "padim"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
CONFIG_PATH = root_directory / f"src/anomalib/models/{MODEL}/config.yaml"
with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
    print(file.read())

# COMMAND ----------

# MAGIC %md
# MAGIC We could use [get_configurable_parameter](https://github.com/openvinotoolkit/anomalib/blob/main/anomalib/config/config.py#L114) function to read the configs from the path and return them in a dictionary. We use the default config file that comes with Padim implementation, which uses `./datasets/MVTec` as the path to the dataset. We need to overwrite this after loading the config.

# COMMAND ----------

# pass the config file to model, callbacks and datamodule
config = get_configurable_parameters(config_path=CONFIG_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset: MVTec AD
# MAGIC
# MAGIC **MVTec AD** is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over **5000** high-resolution images divided into **15** different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects. If the dataset is not located in the root datasets directory, anomalib will automatically install the dataset.
# MAGIC
# MAGIC We could now import the MVtec AD dataset using its specific datamodule implemented in anomalib.

# COMMAND ----------

datamodule = get_datamodule(config)
datamodule.setup()  # Downloads the dataset if it's not in the specified `root` directory
datamodule.prepare_data()  # Create train/val/test/prediction sets.

i, data = next(enumerate(datamodule.val_dataloader()))
print(data.keys())

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check the shapes of the input images and masks.

# COMMAND ----------

print(data["image"].shape, data["mask"].shape)

# COMMAND ----------

# MAGIC %md
# MAGIC We could now visualize a normal and abnormal sample from the validation set.

# COMMAND ----------

def show_image_and_mask(sample: dict[str, Any], index: int) -> Image:
    img = ToPILImage()(Denormalize()(sample["image"][index].clone()))
    msk = ToPILImage()(sample["mask"][index]).convert("RGB")

    return Image.fromarray(np.hstack((np.array(img), np.array(msk))))


# Visualize an image with a mask
show_image_and_mask(data, index=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Model and Callbacks
# MAGIC
# MAGIC Now, the config file is updated as we want. We can now start model training with it. Here we will be using `datamodule`, `model` and `callbacks` to train the model. Callbacks are self-contained objects, which contains non-essential logic. This way we could inject as many callbacks as possible such as ModelLoading, Timer, Metrics, Normalization and Visualization
# MAGIC
# MAGIC In addition to the training, we would like to perform inference using OpenVINO. Therefore we will set the export configuration to openvino so that anomalib would export the trained model to the openvino format.

# COMMAND ----------

# Set the export-mode to OpenVINO to create the OpenVINO IR model.
config.optimization.export_mode = None
#config.optimization.export_mode = "openvino"


# Get the model and callbacks
model = get_model(config)
callbacks = get_callbacks(config)

# COMMAND ----------

model

# COMMAND ----------

from anomalib.post_processing import NormalizationMethod, ThresholdMethod
from pytorch_lightning.callbacks import ModelCheckpoint

from anomalib.data.task_type import TaskType

from anomalib.utils.callbacks import (
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
)
from anomalib.utils.callbacks.export import ExportCallback, ExportMode

callbackstest = [
    MetricsConfigurationCallback(
        task=TaskType.CLASSIFICATION,
        image_metrics=["AUROC"],
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
        filename="model",
        export_mode=ExportMode.OPENVINO,
    ),
]

callbackstest

# COMMAND ----------

callbacks

# COMMAND ----------

import mlflow
#from pytorch_lightning.loggers import MLFlowLogger

username = spark.sql("SELECT current_user()").first()['current_user()']

# When using databricks repos, it is not possible to write into working directories
# specifying a dbfs default dir helps to avoid this
experiment_path = f'/Users/{username}/pytorch-lightning-on-databricks'

# We manually create the experiment so that we know the id and can send that to the worker nodes when we scale
experiment = mlflow.set_experiment(experiment_path)

#Autologging is performed when you call the fit method of pytorch_lightning.Trainer().

# Initialize a trainer
trainer = Trainer(**config.trainer, callbacks=callbacks)

mlflow.pytorch.autolog()

# Train the model
with mlflow.start_run() as run:
  trainer.fit(model=model, datamodule=datamodule)

mlflow.end_run()

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

# load best model from checkpoint before evaluating
load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)

# '/Workspace/Repos/michelle.liu@databricks.com/anomalib/results/padim/mvtec/bottle/run/weights/model-v7.ckpt' for best_model_path

trainer.callbacks.insert(0, load_model_callback)
test_results = trainer.test(model=model, datamodule=datamodule)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Best Model to Registry

# COMMAND ----------

model_name = "Anomalib-Pytorch-Model-1"

model_uri = f"runs:/e765a6bbfbad452b9cae80c65e094789/model"
registered_model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## OpenVINO Inference
# MAGIC Now that we trained and tested a model, we could check a single inference result using OpenVINO inferencer object. This will demonstrate how a trained model could be used for inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load a Test Image
# MAGIC
# MAGIC Let's read an image from the test set and perform inference using OpenVINO inferencer.

# COMMAND ----------

image_path = root_directory / "datasets/MVTec/bottle/test/broken_large/000.png"
image = read_image(path="./datasets/MVTec/bottle/test/broken_large/000.png")
plt.imshow(image)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC Now that we trained and tested a model, we could check a single inference result. This will demonstrate how a trained model could be used for inference.

# COMMAND ----------

from anomalib.data import InferenceDataset
from torch.utils.data import DataLoader

inference_dataset = InferenceDataset(path=image_path, image_size=(256, 256))
inference_dataloader = DataLoader(dataset=inference_dataset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the MLflow Model

# COMMAND ----------

my_model = mlflow.pytorch.load_model("models:/Anomalib-Pytorch-Model-1/1")


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Perform Inference
# MAGIC
# MAGIC Predicting an image using our inferencer is as simple as calling predict method.

# COMMAND ----------

predictions = trainer.predict(model=my_model, dataloaders=inference_dataloader)[0]


# COMMAND ----------

print(predictions.keys())

# COMMAND ----------

print(
    f'Image Shape: {predictions["image"].shape},\n'
    f'Anomaly Map Shape: {predictions["anomaly_maps"].shape}, \n'
    f'Predicted Mask Shape: {predictions["pred_masks"].shape}'
)

# COMMAND ----------

print(predictions['pred_scores'], predictions['pred_labels'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualization
# MAGIC To properly visualize the predictions, we will need to perform some post-processing operations.
# MAGIC
# MAGIC Let's post-process each output one by one. We could start with the image. Each image is a tensor and within (0, 1) range. To visualize it, we need to denormalize it to (0, 255) scale. Anomalib already has a class for this. Let's use it.

# COMMAND ----------


# Visualize the original image
plt.imshow(Denormalize()(predictions['image'][0]))

# COMMAND ----------

# MAGIC %md
# MAGIC The second output of the predictions is the anomaly map. As can be seen above, it's also a torch tensor and of size torch.Size([1, 1, 256, 256]). We therefore need to convert it to numpy and squeeze the dimensions to make it 256x256 output to visualize.

# COMMAND ----------

# Visualize the raw anomaly maps predicted by the model.
plt.imshow(predictions['anomaly_maps'][0].cpu().numpy().squeeze())

# COMMAND ----------


# Visualize the original image
plt.imshow(predictions['pred_masks'][0].cpu().numpy().squeeze())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the OpenVINO Model
# MAGIC
# MAGIC By default, the output files are saved into `results` directory. Let's check where the OpenVINO model is stored.

# COMMAND ----------

output_path = Path(config["project"]["path"])
print(output_path)

# COMMAND ----------

openvino_model_path = output_path / "weights" / "openvino" / "model.bin"
metadata_path = output_path / "weights" / "openvino" / "metadata.json"
print(openvino_model_path.exists(), metadata_path.exists())

# COMMAND ----------

inferencer = OpenVINOInferencer(
    path=openvino_model_path,  # Path to the OpenVINO IR model.
    metadata_path=metadata_path,  # Path to the metadata file.
    device="CPU",  # We would like to run it on an Intel CPU.
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perform Inference
# MAGIC Predicting an image using OpenVINO inferencer is as simple as calling `predict` method.

# COMMAND ----------

print(image.shape)

# COMMAND ----------

predictions = inferencer.predict(image=image)

# COMMAND ----------

# MAGIC %md
# MAGIC where `predictions` contain any relevant information regarding the task type. For example, predictions for a segmentation model could contain image, anomaly maps, predicted scores, labels or masks.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizing Inference Results

# COMMAND ----------

print(predictions.pred_score, predictions.pred_label)

# COMMAND ----------

# Visualize the original image
plt.imshow(predictions.image)

# COMMAND ----------

# Visualize the raw anomaly maps predicted by the model.
plt.imshow(predictions.anomaly_map)

# COMMAND ----------

# Visualize the heatmaps, on which raw anomaly map is overlayed on the original image.
plt.imshow(predictions.heat_map)

# COMMAND ----------

# Visualize the segmentation mask.
plt.imshow(predictions.pred_mask)

# COMMAND ----------

# Visualize the segmentation mask with the original image.
plt.imshow(predictions.segmentations)

# COMMAND ----------

# MAGIC %md
# MAGIC This wraps the `getting_started` notebook. There are a lot more functionalities that could be explored in the library. Please refer to the [documentation](https://openvinotoolkit.github.io/anomalib/) for more details.
