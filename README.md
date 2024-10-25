# NSFW Image Detection

This project aims to create a machine learning model that detects NSFW (Not Safe For Work) images using TensorFlow and Flask. It includes data preprocessing, model training, evaluation, and deployment of a web API.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Model Creation](#model-creation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Using the Transformers Pipeline](#using-the-transformers-pipeline)
- [Creating a Flask App for Deployment](#creating-a-flask-app-for-deployment)
- [Deploying the App](#deploying-the-app)
- [Final Project Structure](#final-project-structure)
- [Acknowledgments](#acknowledgments)

## Requirements

Before you begin, ensure you have the following installed:

- **Python 3.x**: Download from [python.org](https://www.python.org/downloads/).
- **pip**: Comes pre-installed with Python 3.x.

### Libraries

You will need the following Python libraries:

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Flask
- Transformers
- Torch
- torchvision

You can install these libraries using pip. Open a command prompt and run:

```bash
pip install tensorflow numpy pandas matplotlib flask transformers torch torchvision
```

## Setup

1. **Create a project directory:**

   Open a command prompt and run:

   ```bash
   mkdir nsfw_image_detector
   cd nsfw_image_detector
   ```

2. **Create subfolders and files:**

   In the command prompt, run:

   ```bash
   mkdir data\train\data\train\nsfw data\train\sfw
   mkdir data\validation\nsfw data\validation\sfw
   mkdir data\test\nsfw data\test\sfw
   type nul > preprocessing.py
   type nul > model.py
   type nul > train.py
   type nul > evaluate.py
   type nul > app.py
   ```

## Data Preparation

### Organizing Your Data

Organize your images into the following directory structure:

```
data/
├── train/
│   ├── nsfw/  # Add NSFW images here
│   └── sfw/   # Add SFW (Safe for Work) images here
├── validation/
│   ├── nsfw/  # Add NSFW validation images here
│   └── sfw/   # Add SFW validation images here
└── test/
    ├── nsfw/
    └── sfw/
```

### Preprocessing Script

Create a preprocessing script `preprocessing.py` to handle image resizing, augmentation, and data generators.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
validation_dir = 'data/validation'

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

## Model Creation

Create a model script `model.py` to define the architecture using MobileNetV2 (optional) or use the Transformers library for NSFW detection.

### Using the Transformers Library

You can load the model using the Hugging Face `transformers` library.

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
```

## Training the Model

If you decide to train your model using TensorFlow, create a training script `train.py` to train the model.

```python
from preprocessing import train_generator, validation_generator
from model import create_model

model = create_model()

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

model.save('nsfw_detector_model.h5')
```

Run the training script in the command prompt:

```bash
python train.py
```

This will save the trained model as `nsfw_detector_model.h5`.

## Evaluating the Model

Create an evaluation script `evaluate.py` to test the model on the validation dataset.

```python
from tensorflow.keras.models import load_model
from preprocessing import validation_generator

model = load_model('nsfw_detector_model.h5')

loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
```

Run the evaluation script:

```bash
python evaluate.py
```

## Using the Transformers Pipeline

To utilize the Hugging Face pipeline for NSFW image classification:

```python
from transformers import pipeline
from PIL import Image

# Load the image classification pipeline
pipe = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

# Load the image you want to classify
image_path = "path_to_your_image.jpg"  # Update with your image path
image = Image.open(image_path)

# Perform the classification
results = pipe(image)

# Print the results
print(results)
```

## Creating a Flask App for Deployment

Create a Flask app `app.py` to serve your model as an API.

```python
from flask import Flask, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

processor = AutoImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img_path = f'./uploads/{img.filename}'
    img.save(img_path)

    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = logits.argmax(dim=1)

    return jsonify({
        'prediction': 'NSFW' if predictions.item() == 1 else 'SFW'
    })

if __name__ == '__main__':
    app.run(debug=True)
```

Run the Flask app:

```bash
python app.py
```

## Deploying the App

### Deployment Options

You can deploy your Flask app using platforms like:

- **Heroku**: Follow the [Heroku deployment guide](https://devcenter.heroku.com/articles/getting-started-with-python).
- **AWS EC2**: Follow the [AWS EC2 Flask deployment guide](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html).

## Final Project Structure

```
nsfw_image_detector/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── preprocessing.py  # Image preprocessing and data generators
├── model.py          # Model architecture
├── train.py          # Training script
├── evaluate.py       # Model evaluation script
└── app.py            # Flask API for model serving
```

## Acknowledgments

- TensorFlow for providing the necessary libraries and tools.
- The creators of MobileNetV2 for the pre-trained model architecture.
- Hugging Face for the `transformers` library and the `Falconsai/nsfw_image_detection` model.
```
