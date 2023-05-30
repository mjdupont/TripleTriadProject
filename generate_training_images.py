from os import listdir
import os
from os.path import isfile, join
import random
from PIL import Image
import cv2
import numpy as np
from fetch_cards import get_card_filename
import fetch_cards
from constants import BACKGROUNDS_FOLDER, CARD_WIDTH, CARD_HEIGHT, CARD_ZONE_PADDING, GENERATED_FOLDER, SAMPLE_IMAGE_DIR

from distutils.dir_util import copy_tree

from read_screen import BLUE_BACKGROUND_COLOR, GREY_BACKGROUND_COLOR, RED_BACKGROUND_COLOR, color_bounds

IMAGE_WIDTH = CARD_WIDTH + (2*CARD_ZONE_PADDING)
IMAGE_HEIGHT = CARD_HEIGHT + (2*CARD_ZONE_PADDING)

def generate_background_sample(image):
      # Define the size of the desired random subsection
      sub_width = IMAGE_WIDTH
      sub_height = IMAGE_HEIGHT

      # Get the dimensions of the background image
      bg_width, bg_height = image.size

      # Calculate the maximum possible x and y offsets to avoid going out of bounds
      max_x_offset = bg_width - sub_width
      max_y_offset = bg_height - sub_height

      # Generate random x and y offsets within the valid range
      x_offset = random.randint(0, max_x_offset)
      y_offset = random.randint(0, max_y_offset)

      # Crop the random subsection from the background image
      return image.crop((x_offset, y_offset, x_offset + sub_width, y_offset + sub_height))
      
def generate_image_mask(image):
  # Specify range of colors to mask out (fixed range around sampled color)
  grey_lower_bound, grey_upper_bound = color_bounds(GREY_BACKGROUND_COLOR)
  # Remove alpha
  image_vals = np.array(image)[:,:,0:3]
  # Calculate mask of colors in range
  grey_mask = cv2.inRange(image_vals, grey_lower_bound, grey_upper_bound) 
  # Invert where mask found colors in the above range to zeroes, and set ones elsewhere, to subtract values in this range
  mask = np.where(grey_mask == 0, 1, 0)
  return mask

def generate_background_samples(background_sample_folder_path, samples_per_image, generated_backgrounds_path = BACKGROUNDS_FOLDER):
  # Find and load all files from screenshots directory
  background_files = [join(background_sample_folder_path, f) for f in listdir(background_sample_folder_path) if isfile(join(background_sample_folder_path, f))]
  
  # Open and take n samples (random crops of appropriate size) from screenshot
  images = []
  for file in background_files:
    background_image = Image.open(file)
    images = images + [ generate_background_sample(background_image) for _ in range(samples_per_image) ]
  
  # Save all background images
  for i, image in enumerate(images):
    filename = os.path.join(generated_backgrounds_path, f"img{i}.png")
    image.save(filename, "PNG")

def get_card_sample_image_dir(card_datum):
  # Calculate the directory to store training samples
  return join(SAMPLE_IMAGE_DIR, f'card_{card_datum["id"]}')


def generate_training_images(background_sample_folder_path, card_datum, images_per_background):
  # Calculate and if needed create the directory for this card's training data
  card_sample_folder = get_card_sample_image_dir(card_datum)
  if not os.path.exists(card_sample_folder):
    os.makedirs(card_sample_folder)

  # Get all backgrounds to paste masked card image onto
  background_sample_filenames  = [join(background_sample_folder_path, f) for f in listdir(background_sample_folder_path) if isfile(join(background_sample_folder_path, f))]
  
  for i, background_filename in enumerate(background_sample_filenames):
    # Open the background image
    background_image = Image.open(background_filename).convert("RGBA")

    for j in range(images_per_background):
      # Open the image, including alpha
      sample_image = background_image.copy()
      card_image_path = get_card_filename(card_datum)
      card_image = Image.open(card_image_path).convert('RGBA')
      
      # Create a mask from the image
      mask = generate_image_mask(card_image)

      card_image_array = np.array(card_image)
      # Slice only the alpha channel from the image
      card_image_alpha = card_image_array[:,:,3]
      # Multiply the sliced alphase element-wise with the mask; the mask has 0 where the background color was selected,
      # And 1 where the value was, setting background color pixels to zero and preserving the transparency of other pixels
      masked_card_image_alpha = card_image_alpha * mask
      
      # Replace card image's transparency with newly masked alpha values.
      card_image_array[:,:,3] = masked_card_image_alpha

      masked_card_image = Image.fromarray(card_image_array)
      
      # Calculate the random position for placing the image within the background
      x_offset = random.randint(0, 2*CARD_ZONE_PADDING)
      y_offset = random.randint(0, 2*CARD_ZONE_PADDING)
      
      # Overlay the image onto the background at the random position
      sample_image.paste(card_image, (x_offset, y_offset), masked_card_image)
      
      # Save the training sample image
      sample_image_path = os.path.join(card_sample_folder, f"generated_{i}{j}.png")
      sample_image.save(sample_image_path)

def generate_card_back_images(background_sample_folder_path, images_per_background):
  # Calculate and if needed create the directory for this card's training data
  card_sample_folder = join(SAMPLE_IMAGE_DIR, 'unknown_card')
  if not os.path.exists(card_sample_folder):
    os.makedirs(card_sample_folder)

  # Get all backgrounds to paste masked card image onto
  background_sample_filenames  = [join(background_sample_folder_path, f) for f in listdir(background_sample_folder_path) if isfile(join(background_sample_folder_path, f))]
  
  for i, background_filename in enumerate(background_sample_filenames):
    # Open the background image
    background_image = Image.open(background_filename).convert("RGBA")

    for j in range(images_per_background):
      # Open the image, including alpha
      sample_image = background_image.copy()
      card_image_path = 'images/png/card_back.png'
      card_image = Image.open(card_image_path).convert('RGBA')
      
      # Create a mask from the image
      mask = generate_image_mask(card_image)

      card_image_array = np.array(card_image)
      # Slice only the alpha channel from the image
      card_image_alpha = card_image_array[:,:,3]
      # Multiply the sliced alphase element-wise with the mask; the mask has 0 where the background color was selected,
      # And 1 where the value was, setting background color pixels to zero and preserving the transparency of other pixels
      masked_card_image_alpha = card_image_alpha * mask
      
      # Replace card image's transparency with newly masked alpha values.
      card_image_array[:,:,3] = masked_card_image_alpha

      masked_card_image = Image.fromarray(card_image_array)
      
      # Calculate the random position for placing the image within the background
      x_offset = random.randint(0, 2*CARD_ZONE_PADDING)
      y_offset = random.randint(0, 2*CARD_ZONE_PADDING)
      
      # Overlay the image onto the background at the random position
      sample_image.paste(card_image, (x_offset, y_offset), masked_card_image)
      
      # Save the training sample image
      sample_image_path = os.path.join(card_sample_folder, f"generated_{i}{j}.png")
      sample_image.save(sample_image_path)


def generate_all_images(n_backgrounds= 10, training_images_per_background=5):
  if not os.path.exists(BACKGROUNDS_FOLDER):
    # Create the directory
    os.makedirs(BACKGROUNDS_FOLDER)
  
  if not os.path.exists(SAMPLE_IMAGE_DIR):
      # Create the directory
    os.makedirs(SAMPLE_IMAGE_DIR)

  generate_background_samples("background_samples", n_backgrounds, generated_backgrounds_path = BACKGROUNDS_FOLDER)
  generate_random_backgrounds(n_backgrounds, BACKGROUNDS_FOLDER)
  generate_solid_backgrounds(n_backgrounds, BACKGROUNDS_FOLDER)
  card_data = fetch_cards.read_card_list()
  for card_datum in card_data:
    generate_training_images(BACKGROUNDS_FOLDER, card_datum, training_images_per_background)
  generate_card_back_images(BACKGROUNDS_FOLDER, training_images_per_background)

def generate_random_backgrounds(n_backgrounds, output_directory):
  for i in range(n_backgrounds):
    # Generate random image data
    image_data = np.random.randint(0, 256, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    
    # Create a PIL image from the generated data
    image = Image.fromarray(image_data)
    
    # Save the image
    image_path = os.path.join(output_directory, f"static_background_{i}.png")
    image.save(image_path)

def generate_solid_backgrounds(n_random_colors, output_directory, fixed_colors = [BLUE_BACKGROUND_COLOR, GREY_BACKGROUND_COLOR, RED_BACKGROUND_COLOR]):
  # Generate a single solid color for the number of random colors to create
  random_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_random_colors)]

  # Join random and fixed colors
  colors_to_make = random_colors + fixed_colors

  #Create and save a blank background for each of the background colors
  backgrounds = [Image.new("RGBA", (IMAGE_WIDTH, IMAGE_HEIGHT), color) for color in colors_to_make]
  for i, image in enumerate(backgrounds): 
    image_path = os.path.join(output_directory, f"constant_background_{i}.png")
    image.save(image_path)

if (__name__ == "__main__"):
  if not os.path.exists(GENERATED_FOLDER):
    generate_all_images(n_backgrounds = 50, training_images_per_background=1)