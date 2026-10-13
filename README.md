# Kidney CT Scan Classification

A deep-learning project that classifies kidney CT scan images as **Normal** or
**Tumor**. It uses transfer learning with a VGG16 convolutional neural network,
and includes a Flask web interface and a JSON prediction endpoint.

> **Disclaimer:** This project is for educational and research purposes only.
> Its predictions are not a medical diagnosis and must not be used to guide
> clinical decisions.

## What it does

- Downloads and extracts the CT scan dataset from the Google Drive URL in
  `config/config.yaml`.
- Builds a VGG16-based classifier, trains it, and evaluates it on a validation
  split.
- Records evaluation metrics in `scores.json` and logs parameters and metrics
  to the configured MLflow tracking server.
- Serves a small browser interface and an HTTP endpoint for image predictions.

## Getting started

### Requirements

- Python and pip
- Internet access for dependency installation, dataset download, and pretrained
  ImageNet weights

TensorFlow and the other Python dependencies are listed in `requirements.txt`.
TensorFlow installation and performance can vary by operating system and
hardware.

### Install

```bash
git clone https://github.com/kashyap03042003/Kidney-Classification-Project.git
cd Kidney-Classification-Project

python -m venv .venv
```

Activate the virtual environment, then install the dependencies:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Train and evaluate

Run all four pipeline stages in sequence:

```bash
python main.py
```

The stages download the data, prepare the VGG16 base model, train the
classifier, and evaluate it. Outputs include the trained model at
`artifacts/training/model.h5` and evaluation metrics in `scores.json`.

The pipeline is also described in `dvc.yaml` and can be run with DVC:

```bash
dvc repro
```

Training settings such as image dimensions, batch size, epochs, augmentation,
and learning rate are in `params.yaml`. Dataset and model paths are in
`config/config.yaml`.

## Run the web app

Start the Flask application:

```bash
python app.py
```

Open [http://localhost:8080](http://localhost:8080) to use the browser
interface. Upload an image and select **Predict**. The app loads its inference
model from `model/model.h5`.

To serve a newly trained model, copy the training output to the inference path
before starting the app:

```bash
cp artifacts/training/model.h5 model/model.h5
```

On Windows, use `copy artifacts\training\model.h5 model\model.h5`.

## Prediction API

Send a `POST` request to `/predict` with a JSON object containing the image as
a base64-encoded string:

```json
{
  "image": "<base64-encoded image>"
}
```

For example, with `curl`:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64-encoded image>"}'
```

The response is a JSON array containing the predicted class:

```json
[
  {
    "image": "Normal"
  }
]
```

The prediction code resizes input images to 224 × 224 pixels. Send the base64
image data itself, without a `data:image/...;base64,` prefix.

## Project structure

```text
config/                  Dataset and artifact paths
src/kindneyClassifier/   Data, model, training, evaluation, and prediction code
templates/               Flask web interface
artifacts/               Pipeline data and model outputs
model/                   Model used by the prediction app
main.py                  Runs the complete training and evaluation pipeline
app.py                   Starts the Flask web app
params.yaml              Model and training parameters
dvc.yaml                 DVC pipeline stages
```

## License

This project is distributed under the [MIT License](LICENSE).
