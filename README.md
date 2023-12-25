# Transfer-Learning-
Perform image classification using a pre-trained deep learning model (e.g., VGG16) in TensorFlow/Keras.
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

img_path = 'image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Access the top predicted classes and their probabilities
