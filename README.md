

# **NSFW Image Detection Model**

This project demonstrates how to create a machine learning model to detect NSFW (Not Safe for Work) images using Python, TensorFlow/Keras, and Flask. This guide covers steps from setting up the environment, training the model, and deploying it using Flask.

---

### **Prerequisites**
Ensure you have the following installed:
- **Python 3.x**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Pip**: Python package installer (should come with Python)
- **Virtualenv** *(optional but recommended)*: For managing Python packages.

### **Python Libraries**
Install the required libraries using pip:

```bash
pip install tensorflow numpy pandas matplotlib flask
```

---

### **Project Structure**
Set up your project directory with the following structure:

```plaintext
nsfw_image_detector/
│
├── data/
│   ├── train/
│   │   ├── nsfw/   # NSFW images
│   │   └── sfw/    # Safe for Work images
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

---

### **Step 1: Data Preprocessing**
In `preprocessing.py`, preprocess and augment the images for better model training:

```python
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories for training and validation data
train_dir = 'data/train'
validation_dir = 'data/validation'

# Data augmentation for training set
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

# Rescaling for validation set (no augmentation)
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

---

### **Step 2: Build the Model**
In `model.py`, define the model architecture using MobileNetV2 for transfer learning:

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_model():
    # Load MobileNetV2 with pre-trained ImageNet weights
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers (pre-trained part)
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

---

### **Step 3: Train the Model**
In `train.py`, train the model on the preprocessed data:

```python
from preprocessing import train_generator, validation_generator
from model import create_model

# Create the model
model = create_model()

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the trained model
model.save('nsfw_detector_model.h5')
```

Run the training script:

```bash
python train.py
```

---

### **Step 4: Evaluate the Model**
In `evaluate.py`, evaluate model performance on the validation set:

```python
from tensorflow.keras.models import load_model
from preprocessing import validation_generator

# Load the trained model
model = load_model('nsfw_detector_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
```

Run the evaluation:

```bash
python evaluate.py
```

---

### **Step 5: Deploy the Model with Flask**
In `app.py`, create a Flask API to serve the model:

```python
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('nsfw_detector_model.h5')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    img = request.files['image']
    img_path = f'./uploads/{img.filename}'
    img.save(img_path)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model.predict(img_array)[0][0]

    # Return prediction result
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

Test the API using `curl` or Postman:

```bash
curl -X POST -F 'image=@path_to_image.jpg' http://127.0.0.1:5000/predict
```

---

### **Step 6: Deploy to the Cloud**
You can deploy this Flask app to platforms like:
- **Heroku**: [Deploy a Flask App on Heroku](https://devcenter.heroku.com/articles/getting-started-with-python)
- **AWS EC2**: [Deploying Flask on EC2](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html)

---

### **Conclusion**
You have successfully built and deployed an NSFW image detector using Python, TensorFlow, and Flask. To enhance this project, consider:
- Improving the dataset
- Fine-tuning the model with additional layers or different architectures
- Deploying on scalable cloud platforms like AWS or GCP.

