import os
from matplotlib import pyplot as plt
import tensorflow as tf
from constants import CARD_HEIGHT, CARD_WIDTH, CARD_ZONE_PADDING, CHECKPOINT_PATH

from generate_training_images import SAMPLE_IMAGE_DIR

from tensorflow import keras
from read_screen import TF_MODEL_FILE_PATH
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

BATCH_SIZE = 32
IMG_HEIGHT = CARD_HEIGHT + 2 * CARD_ZONE_PADDING
IMG_WIDTH = CARD_WIDTH + 2 * CARD_ZONE_PADDING


DATA_DIR = SAMPLE_IMAGE_DIR

def simple_model(num_classes):

  image_input = keras.layers.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3),name='raw_image')

  x = keras.layers.Rescaling(1./255)(image_input)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)
  x = keras.layers.MaxPooling2D()(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
  x = keras.layers.MaxPooling2D()(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
  x = keras.layers.MaxPooling2D()(x)
  x = keras.layers.Dropout(.2)(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.BatchNormalization()(x)
  #x = keras.layers.Dense(1024, activation='relu')(x)
  x = keras.layers.Dense(512, activation='relu')(x)
  output = keras.layers.Dense(num_classes, name='output')(x)

  model = keras.Model(image_input, output, name='TT_card_classifier')

  model.compile(optimizer='adam',
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), #from_logits=True
                metrics=['accuracy']
                )

  model.summary()

  return model

def alex_net(num_classes):
  
  image_input = keras.layers.Input(shape=(IMG_HEIGHT,IMG_WIDTH,3),name='raw_image')

  x = keras.layers.Rescaling(1./255)(image_input)
  x = keras.layers.Conv2D(96, kernel_size = 7, strides=(3,3), padding='same', activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

#  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Conv2D(256, kernel_size = 5, strides=(2,2), padding='same', activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size = (3,3), strides = (2,2))(x)
#  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Conv2D(384, kernel_size = 3, strides = (1,1), padding='same', activation='relu')(x)
  x = keras.layers.Conv2D(384, kernel_size = 3, strides = (1,1), padding='same', activation='relu')(x)
  x = keras.layers.Conv2D(256, kernel_size = 3, strides = (1,1), padding='same', activation='relu')(x)
  x = keras.layers.MaxPooling2D(pool_size=(3,3))(x)
#  x = keras.layers.Dropout(.1)(x)
  x = keras.layers.Flatten()(x)
# x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Dense(4096, activation='relu')(x)
  x = keras.layers.Dropout(.5)(x)
  x = keras.layers.Dense(4096, activation='relu')(x)
  x = keras.layers.Dropout(.5)(x)
  output = keras.layers.Dense(num_classes, activation = 'softmax', name='output')(x)

  model = keras.Model(image_input, output, name='TT_card_classifier')

  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(), #from_logits=True
                metrics=['accuracy']
                )

  model.summary()

  return model



def main():
    
  train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

  val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

  class_names = train_ds.class_names

  num_classes = len(class_names)

  manual_class_weights = {i:1 for i in range(len(class_names))}
  manual_class_weights.update({len(class_names):200}) # We assume here that we have ~400 classes, and 'none' is likely as prevalent as *any* card in the dataset.

  model =  simple_model(num_classes)

  checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)

  # Create a callback that saves the model's weights every epoch
  cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_dir, 
      verbose=0, 
      save_freq='epoch')


  epochs=10
  history = model.fit(
    train_ds
    , validation_data=val_ds
    , epochs=epochs
    , callbacks=[cp_callback]
    , class_weight=manual_class_weights
  )

  model.save('classification_model.h5')
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

  # Convert the model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the model.
  with open(TF_MODEL_FILE_PATH, 'wb') as f:
    f.write(tflite_model)

if (__name__ == "__main__"):
  main()