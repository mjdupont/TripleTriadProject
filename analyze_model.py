  

import numpy as np
import tensorflow as tf
from categorize_card_model import BATCH_SIZE, DATA_DIR, IMG_HEIGHT, IMG_WIDTH
from PIL import Image
from constants import TF_MODEL_FILE_PATH
import pytesseract

print(pytesseract.image_to_string(Image.open('generated_images\sample_images\card_1\generated_00.png')))
# image_path = 'images/png/card_54.png'
# image_bytes = tf.io.read_file(image_path)
# image = tf.image.decode_image(image_bytes)
# image = tf.cast(image, tf.float32)
# image = tf.expand_dims(image, 0)
# sobel = tf.image.sobel_edges(image)
# sobel_y = np.asarray(sobel[0, :, :, :, 0]) # sobel in y-direction
# sobel_x = np.asarray(sobel[0, :, :, :, 1]) # sobel in x-direction
# sobel_both = (sobel_y[..., 0] / 4 + 0.5)+(sobel_x[..., 0] / 4 + 0.5)
# Image.fromarray(sobel_y[..., 0] / 4 + 0.5).show()
# Image.fromarray(sobel_x[..., 0] / 4 + 0.5).show()
# Image.fromarray(sobel_both).show()
# temp = ()


# interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
# tf_card_classifier = interpreter.get_signature_runner('serving_default')

# val_ds = tf.keras.utils.image_dataset_from_directory(
#   DATA_DIR,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(IMG_HEIGHT, IMG_WIDTH),
#   batch_size=BATCH_SIZE)

# # Initialize empty lists for actual and predicted values
# actual_values = []
# predicted_values = []


# # Iterate over the dataset and collect actual and predicted values
# val_ds_ub = val_ds.unbatch()

# images = list(val_ds_ub.map(lambda x, y: x))
# labels = list(val_ds_ub.map(lambda x, y: y))

# class_names = val_ds.class_names

# for data, label in zip(images, labels):
#     prediction = tf_card_classifier(rescaling_input=np.expand_dims(data,0))['dense_1']
#     actual_values.append(class_names[label.numpy()])  # Assuming labels are stored as tensors
#     predicted_values.append(class_names[np.argmax(prediction)])  # Assuming `model` is your trained model

# # Convert lists to NumPy arrays if desired
# actual_values = np.array(actual_values)
# predicted_values = np.array(predicted_values)

# # Print the actual and predicted values
# print("Actual Values:", actual_values)
# print("Predicted Values:", predicted_values)

