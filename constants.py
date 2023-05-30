from os.path import join

CARD_HEIGHT = 128
CARD_WIDTH = 104
CARD_ZONE_PADDING = 20

MODEL_FOLDER = 'model'

GENERATED_FOLDER = 'generated_images'
SAMPLE_IMAGE_DIR = join(GENERATED_FOLDER, 'sample_images')
BACKGROUNDS_FOLDER = join(GENERATED_FOLDER, join('sample_images', 'no_card'))

TF_MODEL_FILE_PATH = join(MODEL_FOLDER, 'card_recognition_model.tflite')

CHECKPOINT_PATH = join(MODEL_FOLDER, join("model_checkpoints,cp-{epoch:04d}.ckpt"))