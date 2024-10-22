
```markdown
# NSFW Image Detection Model

This project demonstrates how to create a machine learning model to detect NSFW (Not Safe for Work) images using Python, TensorFlow/Keras, and Flask. This guide covers all steps from setting up the environment, training the model, and deploying it using Flask.

## Prerequisites

Ensure you have the following installed:
- **Python 3.x** ([Download here](https://www.python.org/downloads/))
- **Git** ([Download here](https://git-scm.com/downloads))
- **Pip** (Python package installer)
- **Virtualenv** (optional, for package management)

### Python Libraries
Install the required libraries using:
```bash
pip install tensorflow numpy pandas matplotlib flask
```

## Project Structure

Set up your project directory:
```bash
nsfw_image_detector/
│
├── data/
│   ├── train/
│   │   ├── nsfw/  # Contains NSFW images
│   │   └── sfw/   # Contains SFW images
│   ├── validation/
│   │   ├── nsfw/
│   │   └── sfw/
│   └── test/
│       ├── nsfw/
│       └── sfw/
├── preprocessing.py
├── model.py
├── train.py
├── evaluate.py
└── app.py
```

## Step 1: Data Preprocessing

In `preprocessing.py`, preprocess and augment the images:
```python
import os
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

## Step 2: Build the Model

In `model.py`, define the model architecture using MobileNetV2:
```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

## Step 3: Train the Model

In `train.py`, train the model:
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

Run the training script:
```bash
python train.py
```

## Step 4: Evaluate the Model

In `evaluate.py`, evaluate model performance:
```python
from tensorflow.keras.models import load_model
from preprocessing import validation_generator

model = load_model('nsfw_detector_model.h5')

loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
```

Run the evaluation:
```bash
python evaluate.py
```

## Step 5: Deploy the Model with Flask

In `app.py`, create a simple Flask API to serve the model:
```python
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('nsfw_detector_model.h5')
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img_path = f'./uploads/{img.filename}'
    img.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    prediction = model.predict(img_array)[0][0]

    return jsonify({
        'prediction': 'NSFW' if prediction > 0.5 else 'SFW',
        'confidence': float(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
```

Run the Flask app:
```bash
python app.py
```

Test the API using **curl** or **Postman**:
```bash
curl -X POST -F 'image=@path_to_image.jpg' http://127.0.0.1:5000/predict
```

## Step 6: Deploy to the Cloud

You can deploy the Flask app to platforms like:
- **Heroku**: [Deploy a Flask App on Heroku](https://devcenter.heroku.com/articles/getting-started-with-python#introduction)
- **AWS EC2**: [Deploying Flask on EC2](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html)

## Conclusion

You have successfully built and deployed an NSFW image detector. You can enhance this project by improving the dataset, fine-tuning the model, and deploying on scalable cloud platforms.

---
**Note:** This project uses a pre-trained MobileNetV2 model for binary classification of NSFW and SFW images.
```
