# %% [markdown] id="blOJ1buw83T9"
# # Envisioning Dante - Graphics Detector

# %% [markdown] id ="yLERLQAN3pM0"
# ## 1 - Read Me First
#
# This project is a [Jupyter](https://jupyter.org/) notebook and was
# designed to run in [Google
# Colab](https://colab.research.google.com/).  If you are not reading
# this notebook in Google Colab, click
# [here](https://colab.research.google.com/github/ox-vgg/demo-notebooks/blob/main/detectors/envdante-detector.ipynb).

# %% [markdown] id="-tfDPTizFHUi"
# ### 1.1 - What is, and how to use, a Jupyter notebook
#
# A Jupyter notebook is a series of "cells".  Each cell contains
# either text (like this one) or code (like others below).  A cell
# that contains code will have a "Run cell" button on the left side
# like this "<img height="18rem" alt="The 'Run cell' button in Colab"
# src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAwCAQAAAD9CzEMAAABTklEQVRYw+2XMU7DMBRAX6ss3VA7VV25AFNWzsDQXoAzVDlBKw6QDJwhTO3OCVjaka0VXVKJDUVC4jOgiMHYcRx9S0j9f7XfS5x8+xsu8R9iQEpGyY4TgnBiR0lGyqA/fMaaI2LJI2tm4fAxObUV3mRNzjgEP+fcCm/yzLwbPKHwhjdZkPjiR2w64wVhw8jv6bdBeEHY+rxFEYz/WaiWWPTCC8LChZ9Q9RZUTOyCvDdeEHJ71drL6o43b0Ftq+6VYxJc8ciXp2L1F37IwSkAuOXVS3BgaApS55TfInzg00ORmoLMSwBww0urIDMFpbcAEpZ8OMeXpmDfQQBwzbNj/N6cUHUUANzzbi03I+oAAUx5stRCfIH6Eql/ZPXfVL3Q1LcK9c1OfbuOcOCoH5kRDn31tiVC4xWhdVRvfiO07xEuIFGuUBEugVGusZfQj28NImRviDLNnQAAAABJRU5ErkJggg==">".
# When you click the "Run cell" button, the code in that cell will run
# and when it finishes, a green check mark appears next to the "Run
# cell" button".  You need to wait for the code in that cell to finish
# before "running" the next cell.

# %% [markdown] id="4kkvU97kDjYh"
# ### 1.2 - Particulars of this notebook
#
# This notebook was designed to run in Google Colab and to analyse
# images in Google Drive.  As such, it requires a Google account.
#
# You must run the cells on this notebook one after the other since
# each cell is dependent on the results of the previous cell.

# %% [markdown] id="yLHvmZjF5HWk"
# ### 1.3 - GPU access
#
# A GPU is not required to run this program but without a GPU it will
# run much slower.  Depending on the amount of data to analyse, it
# might not be sensible to use it without a GPU>
#
# By default, this notebook will run with a GPU.  However, it is
# possible that you were not allocated one, typically because you've
# used up all your GPU resources.  You can confirm this, and possibly
# change it, manually.  To do that, navigate to "Edit" -> "Notebook
# Settings" and select "GPU" from the "Hardware Accelerator" menu.

# %% [markdown] id="y7ZIaFAD1Org"
# ## 2 - Setup

# %% [markdown] id="tKauGGpqIi83"
# ### 2.1 - Check for GPU access

# %% cellView="form" id="5h91qABuhv__"
#@markdown By default, this notebook will run with a GPU.  However, it
#@markdown is possible that you were not allocated one.  If you get a
#@markdown message saying that you do not have access to a GPU,
#@markdown navigate to "Edit" -> "Notebook Settings" and select "GPU"
#@markdown from the "Hardware Accelerator" menu.  If you change it,
#@markdown you need to run this cell again.

# We do this before everything else, namely before installing
# detectron2 (which takes a lot of time), to identify early the case
# of accidentally running this without a GPU.
import torch.cuda

if torch.cuda.is_available():
    USE_GPU = True
    print("You are using this GPU:")
    print(
        "GPU %d: %s (%d GB)"
        % (
            torch.cuda.current_device(),
            torch.cuda.get_device_name(),
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
            * 1e-9,
        )
    )
else:
    USE_GPU = False
    print("You are NOT connected to a GPU")
    print("Consider reconnecting to a runtime with GPU access.")

# %% [markdown] id="U744F4c3s2Yp"
# ### 2.2 - Install dependencies

# %% cellView="form" id="HYJHIkdn3Z1J"
#@markdown This step can take a few of minutes to finish.

# Detectron2 is not available on PyPI, we have to install it from
# their git repos.

# Using `pip install --quiet` is not enough, it still prints out a
# mysterious "Preparing metadata (setup.py)" message which is why we
# redirect stdout to `/dev/null`.  Important messages should go to
# stderr anyway.
# !pip install --quiet git+https://github.com/facebookresearch/detectron2.git > /dev/null

# %% [markdown] id="rmMI8QbZVnUf"
# ### 2.3 - Settings

# %% cellView="form" id="hF_7RBxmDCTx"
#@markdown When the model detects something, that detection is
#@markdown made with a confidence score between 0 and 100%.
#@markdown Detections with a confidence score lower than the selected
#@markdown threshold will be discarded.

CONFIDENCE_THRESHOLD = 50  #@param {type: "slider", min: 0, max: 100, step: 1}
CONFIDENCE_THRESHOLD /= 100.0

# In the future we can make these options but at the moment we only have
# these anyway.
DETECTRON2_CONFIG = "https://thor.robots.ox.ac.uk/staging/env-dante/mask-rcnn-R-50-FPN-D526v2-2024-03-12.py"
MODEL_CKPT = "https://thor.robots.ox.ac.uk/staging/env-dante/mask-rcnn-R-50-FPN-D526v2-2024-03-12.pth"


# %% [markdown] id="hF_7RBxmDCTx"
# ### 2.4 - Load dependencies and configure

# %% cellView="form" id="iimL6Db6gmbb"

#@markdown This cell prepares the detector to run.  This is the place
#@markdown to make changes to the code if you want (but you should not
#@markdown need to).

import logging

import PIL.Image
import detectron2.checkpoint
import detectron2.config
import detectron2.data
import detectron2.data.catalog
import detectron2.data.detection_utils
import detectron2.data.transforms
import detectron2.structures.masks
import detectron2.utils.visualizer
import numpy as np
import torch

import google.colab.output
import google.colab.files
from google.colab.patches import cv2_imshow


_logger = logging.getLogger()
logging.basicConfig()


class Predictor:
    """Simple end to end detection predictor given a LazyConfig."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = detectron2.config.instantiate(cfg.model)
        self.model = self.model.to(cfg.train.device)
        checkpointer = detectron2.checkpoint.DetectionCheckpointer(self.model)
        checkpointer.load(cfg.train.init_checkpoint)
        self.augmentations = detectron2.data.transforms.AugmentationList(
            [
                detectron2.config.instantiate(x)
                for x in cfg.dataloader.test.mapper.augmentations
            ]
        )
        self.model.eval()

    def __call__(self, original_image):
        with torch.no_grad():
            # Apply pre-processing to image.
            height, width = original_image.shape[:2]
            image = self.augmentations(
                detectron2.data.transforms.AugInput(original_image)
            ).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def build_thing_colours(metadata, thing_name_to_colour_name):
    thing_colours = {}
    for i, thing_name in enumerate(metadata.thing_classes):
        colour_name = thing_name_to_colour_name[thing_name]
        rgb = tuple([x / 255.0 for x in PIL.ImageColor.getrgb(colour_name)])
        thing_colours[i] = rgb
    return thing_colours


def pred_classes_to_colours(pred_classes, metadata):
    return [metadata.thing_colors[i] for i in pred_classes.tolist()]


def pred_classes_to_labels(pred_classes, metadata):
    return [metadata.thing_classes[i] for i in pred_classes.tolist()]


def pred_boxes_to_masks(boxes):
    masks = []
    for box in np.asarray(boxes.to("cpu")):
        masks.append([np.array([
            box[0], box[3],
            box[2], box[3],
            box[2], box[1],
            box[0], box[1]
        ])])
    return detectron2.structures.masks.PolygonMasks(masks)


def show_instance_predictions(img, predictions, metadata, score_thresh):
    v = detectron2.utils.visualizer.Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        instance_mode=detectron2.utils.visualizer.ColorMode.SEGMENTATION,
    )
    wanted = predictions["instances"].scores > score_thresh
    out = v.overlay_instances(
        boxes=predictions["instances"].pred_boxes[wanted].to("cpu"),
        masks=pred_boxes_to_masks(predictions["instances"].pred_boxes[wanted]),
        labels=pred_classes_to_labels(
            predictions["instances"].pred_classes[wanted], metadata
        ),
        assigned_colors=pred_classes_to_colours(
            predictions["instances"].pred_classes[wanted], metadata
        )
    )
    cv2_imshow(out.get_image()[:, :, ::-1])


cfg = detectron2.config.LazyConfig.load(
    detectron2.utils.file_io.PathManager.get_local_path(DETECTRON2_CONFIG)
)
cfg.train.init_checkpoint = MODEL_CKPT
if USE_GPU:
    cfg.train.device = "cuda"
else:
    cfg.train.device = "cpu"

metadata = detectron2.data.catalog.MetadataCatalog.get(cfg.dataloader.test.dataset.names[0])

metadata.set(
    thing_colors=build_thing_colours(
        metadata,
        {
            "graphic": "blue",
            "initial-capital": "magenta",
            "manicules": "lime",
            "page-number": "purple",
            "poem": "green",
            "running-header": "red",
            "section-header": "orange",
            "sideletter": "brown",
            "sidenote": "yellow",
            "unpainted-guideletter": "violet",
            "catchword-signature": "cyan",
        }
    )
)

predictor = Predictor(cfg)


# %% [markdown] id="AXRwfL1NAw6O"
# ## 3 - Run Detector

# %% [markdown] id="H4Rw3Bv7RBYi"
# ### 3.1 - Upload Images and Run Detector

# %% cellView="form" id="0FDT_4TUPzzW"
#@markdown When you run this cell, a "Browse..." button will appear at
#@markdown the bottom of the cell.  When you press it, a dialog to
#@markdown upload files will appear.  Select any number of images.
#@markdown When all selected images finish uploading, they will be
#@markdown evaluated one at a time, and the detection results
#@markdown displayed.

google.colab.output.no_vertical_scroll()

uploaded = google.colab.files.upload()
for fpath in uploaded.keys():
    try:
        img = detectron2.data.detection_utils.read_image(
            fpath,
            cfg.dataloader.train.mapper.image_format
        )
    except Exception as exc:
        _logger.error("Failed to read %s: %s", fpath, exc)
    predictions = predictor(img)
    show_instance_predictions(
        img, predictions, metadata, CONFIDENCE_THRESHOLD
    )
