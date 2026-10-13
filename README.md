# Kidney CT Scan Classification

A deep-learning application for classifying kidney CT scan images as **Normal**
or **Tumor**. The project uses transfer learning with a VGG16 image-classification
model and includes a configurable training pipeline, model evaluation, experiment
tracking, and a Flask interface for image predictions.

> **Medical disclaimer:** This project is intended for educational and research
> purposes only. It is not a medical device, has not been validated for clinical
> use, and must not be used to diagnose, treat, or make decisions about a
> patient's health. Always consult qualified medical professionals.

## Overview

The repository brings together the main steps in an image-classification
workflow:

1. **Data ingestion** downloads and extracts the configured CT scan dataset.
2. **Model preparation** creates a VGG16 base model with ImageNet weights and
   adds a two-class classification head.
3. **Training** trains the classifier using a train/validation split, with
   optional image augmentation.
4. **Evaluation** measures the trained model and saves loss and accuracy.
5. **Prediction** serves a trained model through a browser UI and a JSON API.

The pipeline is implemented as separate stages and is also described in
`dvc.yaml`. Training options are in `params.yaml`; data, model, and artifact
paths are in `config/config.yaml`.

## Model and data

The model uses VGG16 as a feature extractor with ImageNet-pretrained weights.
The original classification layer is omitted and replaced with a softmax
classification head. Images are configured at 224 × 224 pixels, and the
training data loader uses a 20% validation split. When enabled, training
augmentation applies random flips, rotations, zooms, and translations.

The dataset archive is downloaded from the Google Drive URL configured in
`config/config.yaml` and is expected to contain a `kidney-ct-scan-image`
directory. The training and evaluation code loads image classes from
subdirectories in that directory.

Default parameters are defined in `params.yaml`:

| Parameter | Default | Description |
| --- | --- | --- |
| `IMAGE_SIZE` | `[224, 224, 3]` | Model input dimensions |
| `BATCH_SIZE` | `16` | Number of images per training batch |
| `EPOCHS` | `1` | Training epochs |
| `CLASSES` | `2` | Number of model output classes |
| `WEIGHTS` | `imagenet` | VGG16 pretrained weights |
| `INCLUDE_TOP` | `False` | Excludes VGG16's original classifier |
| `LEARNING_RATE` | `0.01` | SGD optimizer learning rate |
| `AUGMENTATION` | `True` | Enables training image augmentation |

These are repository defaults, not a statement about model performance or
clinical suitability.

## Requirements

- Python and pip
- Internet access to install dependencies, download the dataset, and retrieve
  pretrained ImageNet weights
- A system capable of running TensorFlow

Dependencies are listed in `requirements.txt`. TensorFlow installation and
hardware acceleration vary by platform; use a TensorFlow-supported Python and
operating-system combination.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/kashyap03042003/Kidney-Classification-Project.git
cd Kidney-Classification-Project
python -m venv .venv
```

Activate the environment and install the project dependencies:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Train and evaluate

Run the complete pipeline from the repository root:

```bash
python main.py
```

The script runs the data-ingestion, base-model preparation, training, and
evaluation stages in order. It downloads the dataset before training, so ensure
the configured data source is accessible.

Alternatively, run the stages through DVC:

```bash
dvc repro
```

Expected outputs include:

- Extracted dataset under `artifacts/data_ingestion/`
- Prepared base model under `artifacts/prepare_base_model/`
- Trained model at `artifacts/training/model.h5`
- Evaluation metrics in `scores.json`

Evaluation also logs parameters and metrics to MLflow. The tracking URI defaults
to the project's configured DagsHub tracking server and can be overridden with
the `MLFLOW_TRACKING_URI` environment variable.

## Run the prediction app

The Flask app loads its inference model from `model/model.h5`. After training,
copy the resulting model into that location:

```bash
cp artifacts/training/model.h5 model/model.h5
```

On Windows:

```powershell
copy artifacts\training\model.h5 model\model.h5
```

Start the app from the repository root:

```bash
python app.py
```

Open [http://localhost:8080](http://localhost:8080) to use the browser
interface. Upload an image and select **Predict**.

The app's built-in Flask server is intended for local development and
demonstration, not production deployment.

## Prediction API

The app provides a `POST /predict` endpoint. Send a JSON body with an `image`
field containing the base64-encoded image data:

```json
{
  "image": "<base64-encoded image data>"
}
```

Example request:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64-encoded image data>"}'
```

The response contains the predicted label:

```json
[
  {
    "image": "Normal"
  }
]
```

The prediction pipeline resizes images to 224 × 224 pixels. Provide the
base64-encoded image data without a `data:image/...;base64,` prefix.

## Repository structure

```text
config/
  config.yaml                 Dataset and artifact locations
src/kindneyClassifier/
  components/                 Data ingestion, model, training, evaluation
  pipeline/                   Pipeline stages and prediction pipeline
  config/                     Configuration loading
templates/
  index.html                  Flask browser interface
artifacts/                    Dataset and generated pipeline artifacts
model/
  model.h5                    Model loaded by the prediction app
main.py                       Runs all training and evaluation stages
app.py                         Starts the Flask application
params.yaml                   Model and training parameters
dvc.yaml                      DVC pipeline definition
requirements.txt              Python dependencies
scores.json                   Evaluation metrics
```

## License

This project is licensed under the [MIT License](LICENSE).
