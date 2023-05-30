from functools import partial
import PIL
import cv2
import numpy as np
import time
from PIL import ImageGrab, Image
from screeninfo import get_monitors
import os

import tensorflow as tf
from constants import CARD_HEIGHT, CARD_WIDTH, CARD_ZONE_PADDING, SAMPLE_IMAGE_DIR, TF_MODEL_FILE_PATH
import fetch_cards
from math import dist
import matplotlib.pyplot as plt
import cProfile
import multiprocessing
import keyboard

PARALLEL = True
USE_MASK = True
MASK_THRESHOLD = .8
UNMASKED_THRESHOLD = .5

DISTANCE_THRESHOLD = 100

METHOD = cv2.TM_CCOEFF_NORMED

RED_BACKGROUND_COLOR = (122,57,58)
BLUE_BACKGROUND_COLOR = (53,86,117)
GREY_BACKGROUND_COLOR = (90,87,90)


RIGHT_HAND_1 = (198,810)
RIGHT_HAND_2 = (198,918)
RIGHT_HAND_3 = (198,1026)
RIGHT_HAND_4 = (336,864)
RIGHT_HAND_5 = (336,972)

LEFT_HAND_1 = (198,40)
LEFT_HAND_2 = (198,148)
LEFT_HAND_3 = (198,256)
LEFT_HAND_4 = (336,94)
LEFT_HAND_5 = (336,202)


HAND_X_OFFSET = 108
HAND_SECOND_ROW_X_OFFSET = 54
HAND_Y_OFFSET = 138
HAND_TO_HAND_X_OFFSET = 770
HAND_TO_BOARD_REF_X_OFFSET = 358
HAND_TO_BOARD_REF_Y_OFFSET = -138

BOARD_X_OFFSET = 136
BOARD_Y_OFFSET = 136

# https://stackoverflow.com/a/31171430
def get_main_monitor():
  monitors = get_monitors()
  primary_monitor = [m for m in monitors if m.is_primary == True][0]
  return primary_monitor

def calculate_bounding_box(monitor, x_perc, y_perc, x_adj, y_adj):
  x_border_width = (monitor.width - (monitor.width * x_perc))/2
  y_border_height = (monitor.height - (monitor.height * y_perc))/2
  return (x_border_width + x_adj, y_border_height+ y_adj, (monitor.width-x_border_width)+x_adj, (monitor.height-y_border_height)+y_adj)

def calculate_hand_position_from_row_and_col(reference_point, row, col):
  y,x = reference_point
  return (y + (row * HAND_Y_OFFSET), x + (row * HAND_SECOND_ROW_X_OFFSET) + (col * HAND_X_OFFSET))

def calculate_hand_from_reference(reference_point, side='left'):
  return {
    f'{side}_hand_1': calculate_hand_position_from_row_and_col(reference_point, 0, 0),
    f'{side}_hand_2': calculate_hand_position_from_row_and_col(reference_point, 0, 1),
    f'{side}_hand_3': calculate_hand_position_from_row_and_col(reference_point, 0, 2),
    f'{side}_hand_4': calculate_hand_position_from_row_and_col(reference_point, 1, 0),
    f'{side}_hand_5': calculate_hand_position_from_row_and_col(reference_point, 1, 1),
  }

def calculate_board_position_from_row_and_col(reference_point, row, col):
  y,x = reference_point
  return (y + (row * BOARD_Y_OFFSET), x + (col * BOARD_X_OFFSET))

def calculate_board_from_reference(reference_point):
  return {
    'UL': calculate_board_position_from_row_and_col(reference_point, 0, 0),
    'UM': calculate_board_position_from_row_and_col(reference_point, 0, 1),
    'UR': calculate_board_position_from_row_and_col(reference_point, 0, 2),
    'ML': calculate_board_position_from_row_and_col(reference_point, 1, 0),
    'MM': calculate_board_position_from_row_and_col(reference_point, 1, 1),
    'MR': calculate_board_position_from_row_and_col(reference_point, 1, 2),
    'LL': calculate_board_position_from_row_and_col(reference_point, 2, 0),
    'LM': calculate_board_position_from_row_and_col(reference_point, 2, 1),
    'LR': calculate_board_position_from_row_and_col(reference_point, 2, 2),
  }

def calculate_card_locations_from_reference(reference_point):
  y,x = reference_point
  right_hand_reference = (y, x + HAND_TO_HAND_X_OFFSET)
  board_reference_point = (y + HAND_TO_BOARD_REF_Y_OFFSET, x + HAND_TO_BOARD_REF_X_OFFSET)
  return {
    'left_hand' : calculate_hand_from_reference(reference_point, side='left'),
    'right_hand' : calculate_hand_from_reference(right_hand_reference, side='right'),
    'board' : calculate_board_from_reference(board_reference_point)
  }

def zip_score(matching_image, x, y):
  return zip(x, y, matching_image[x,y])

def expanded_card_rectangle(point):
  return ((point[1] - CARD_ZONE_PADDING, point[0] - CARD_ZONE_PADDING, point[1] + CARD_WIDTH + CARD_ZONE_PADDING, point[0] + CARD_HEIGHT + CARD_ZONE_PADDING))

def card_rectangle(point):
  return ((point[1], point[0]), (point[1] + CARD_WIDTH, point[0] + CARD_HEIGHT))

def draw_rectangle(image, point, color = (0, 0, 255)):
  cv2.rectangle(image, *card_rectangle(point), color, 1)

def rgbdiff(col_1, col_2):
  return (col_1[0] - col_2[0], col_1[1] - col_2[1], col_1[2] - col_2[2])

def color_bounds(color):
  return rgbdiff(color, (20,20,20)), rgbdiff(color, (-20,-20,-20))

def detect_color(screen, point, template):
  ul, lr = card_rectangle(point)
  cropped = screen.crop((*ul, *lr))
  cropped_vals = np.array(cropped)
  template_vals = template[:,:,0:3]

  red_lower_bound, red_upper_bound = color_bounds(RED_BACKGROUND_COLOR)
  blue_lower_bound, blue_upper_bound = color_bounds(BLUE_BACKGROUND_COLOR)
  grey_lower_bound, grey_upper_bound = color_bounds(GREY_BACKGROUND_COLOR)

  red_mask = cv2.inRange(cropped_vals, red_lower_bound, red_upper_bound)
  blue_mask = cv2.inRange(cropped_vals, blue_lower_bound, blue_upper_bound)
  grey_mask = cv2.inRange(template_vals, grey_lower_bound, grey_upper_bound)

  if cv2.countNonZero(red_mask * grey_mask) > cv2.countNonZero(blue_mask * grey_mask):
    return 'red'
  else:
    return 'blue'
  
def choose_color(color):
  if color == 'red' : 
    return BLUE_BACKGROUND_COLOR 
  else: 
    return RED_BACKGROUND_COLOR

def match_template(screen, screen_image, template_pair):
  card_data, template = template_pair
  results = []
  template_image = template[:,:,0:3]
  if USE_MASK:
    mask = template[:,:,3]
    mask = np.uint8(mask == 255)
    matching_image = cv2.matchTemplate(screen, template_image, METHOD, None, mask=mask)
    threshold = MASK_THRESHOLD
  else:
    matching_image = cv2.matchTemplate(screen, template_image, METHOD)
    threshold = UNMASKED_THRESHOLD
  loc = np.where(matching_image >= threshold)
  if np.any(loc):
    # Zip scores in, needed for finding "best" match, and considering if two cards exist.
    zipped_scores = list(zip_score(matching_image, *loc))
    # Identify "best" match
    max_point = max(zipped_scores, key=lambda tuple: tuple[2])
    max_point_color = detect_color(screen_image, max_point, template)
    #draw_rectangle(screen, max_point, choose_color(max_point_color))
    output = card_data, max_point, max_point_color
    results = results + [output]

    # Remove "nearby" matches
    second_zipped_scores = [(y,x,score) for (y,x,score) in zipped_scores if dist((y,x),(max_point[0], max_point[1])) > DISTANCE_THRESHOLD]
    if any(second_zipped_scores):
      second_max_point = max(second_zipped_scores, key=lambda tuple: tuple[2])
      second_max_point_color = detect_color(screen_image, second_max_point, template)
      #draw_rectangle(screen, second_max_point, choose_color(second_max_point_color))
      output_2 = card_data, second_max_point, second_max_point_color
      results = results + [output_2]
  
  return results

def map_dict(func, d):
    if isinstance(d, dict):
        return {k: map_dict(func, v) for k, v in d.items()}
    else:
        return func(d)

def find_card_positions_from_reference_card(reference_card, show_locations=False):
  screen_image = ImageGrab.grab()
  screen = np.array(screen_image)
  screen = cv2.cvtColor(src=screen, code=cv2.COLOR_BGR2RGB) # Needed for match_template
  reference_template = cv2.imread(fetch_cards.get_card_filename(reference_card), cv2.IMREAD_UNCHANGED) 
  reference = (reference_card, reference_template)
  template_matches = match_template(screen, screen_image, reference)
  template_match = min(template_matches, key= lambda triple: triple[1][1] )
  (_matched_template, (y, x, _score), _color) = template_match
  reference_point = (y,x)
  card_locations = calculate_card_locations_from_reference(reference_point)
  card_boxes = map_dict(expanded_card_rectangle, card_locations)
  
  if show_locations:
    for section,boxes in card_boxes.items():
      for box_name,bbox in boxes.items():
          screen_image = ImageGrab.grab(bbox=bbox)
          screen = np.array(screen_image)
          screen = cv2.cvtColor(src=screen, code=cv2.COLOR_BGR2RGB) # Needed for match_template
          cv2.imshow(f'{box_name}', screen)
          cv2.waitKey(0)
    cv2.destroyAllWindows()
  
  return card_boxes

def card_values(card_data):
  return tuple(card_data['stats']['numeric'].values())

def adjacent_values(val, card_vals):
  adjacent_indices = \
    [ (0,1)
    , (1,2)
    , (2,3)
    , (3,0)
    ]
  adjacent_values = [ card_vals[a] >= val and card_vals[b] >= val for a,b in adjacent_indices ]
  return any(adjacent_values)

def stat_ordering(card_data, max_val):
  card_vals = card_values(card_data)
  if adjacent_values(max_val, card_vals) : return 1
  elif any(np.array(card_vals) >= max_val) : return 2
  else: return 3

def card_data_sort_order(card_data):
  match card_data['stars']:
    case 3: return 1, stat_ordering(card_data, 8)
    case 5: return 2, stat_ordering(card_data, 10)
    case 4: return 3, stat_ordering(card_data, 9)
    case 2: return 4, stat_ordering(card_data, 7)
    case 1: return 5, stat_ordering(card_data, 6)
    

def run_program(mp_pool = None):
  all_card_data = fetch_cards.read_card_list()
  ordered_card_data = sorted(all_card_data, key=card_data_sort_order)
  templates = [None] * len(all_card_data)
  for i, card_data in enumerate(ordered_card_data):
    file_name = fetch_cards.get_card_filename(card_data)
    template = cv2.imread(file_name, cv2.IMREAD_UNCHANGED) 
    templates[i] = (card_data, template)

  card_boxes = find_card_positions_from_reference_card([card_data for card_data in all_card_data if card_data['name'] == 'Titan'][0])

  interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
  tf_card_classifier = interpreter.get_signature_runner('serving_default')

  class_names = [entry.name for entry in os.scandir(SAMPLE_IMAGE_DIR) if entry.is_dir() and entry.name != 'no_card' and entry.name != 'unknown_card']
  card_id_from_class_names = [int(class_name.split('_')[1]) for class_name in class_names ]

  for section,boxes in card_boxes.items():
    for box_name,bbox in boxes.items():
      screen_image = ImageGrab.grab(bbox=bbox)
      screen_image.show("original")
      screen = np.array(screen_image)
      screen = cv2.cvtColor(src=screen, code=cv2.COLOR_BGR2RGB) # Needed for match_template

      img_array = tf.keras.utils.img_to_array(screen_image)
      img_array = tf.expand_dims(img_array, 0) # Create a batch

      match = None
      if PARALLEL:
        with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as match_pool:
          predictions_lite = tf_card_classifier(raw_image=img_array)['output'][0]

          # numpy = predictions_lite.numpy()[0]
          ordered_cards = list(reversed(sorted(enumerate(predictions_lite), key=lambda tup: tup[1])))
          card_id_idx = np.argmax(predictions_lite)

          if card_id_idx == (len(card_id_from_class_names)) :
            card_name = 'no_card'
          elif card_id_idx == (len(card_id_from_class_names) + 1) :
            card_name = 'unknown_card'
          else:
            card_id = card_id_from_class_names[card_id_idx]
            card_datum = [card_datum for card_datum in all_card_data if card_datum['id'] == card_id][0]
            card_name = card_datum['name']

          print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(card_name, 100 * np.max(predictions_lite))
          )

          function = partial(match_template, screen, screen_image)
          results = match_pool.imap_unordered(function, templates)
          for result in results:
            if len(result) > 0:
              match = result[0]
              match_pool.terminate()
              break
      else:
        results = [None] * len(templates)
        for i, template in enumerate(templates):
          result = match_template(screen, screen_image, template)
          if len(result) > 0:
            match = result[0]
            break
      
      if match is not None:
        card_data, (max_point_x, max_point_y, score), color = match
        print(box_name, card_data['name'])
      else: print(box_name, 'none')

  should_continue = True

  # See:
  # https://stackoverflow.com/a/35378944
  # https://towardsdatascience.com/image-analysis-for-beginners-how-to-read-images-video-webcam-and-screen-3778e26760e2
  while(should_continue):
    print("\n***************************************************************************\n")
    start_time = time.time()
    bounding_box = calculate_bounding_box(get_main_monitor(), .60, .48, 47, -38)
    # bounding_box_left_hand = (425,400,775,725)
    # bounding_box_right_hand = (1185,400,1535,725)

    # calculate_card_locations_from_reference(195,5)
    screen_image = ImageGrab.grab(bbox=bounding_box)
    screen = np.array(screen_image)
    screen = cv2.cvtColor(src=screen, code=cv2.COLOR_BGR2RGB)
    
    if mp_pool is not None:
      with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as match_pool:
        function = partial(match_template, screen, screen_image)
        results = sum(match_pool.map(function, templates), [])
    else:
      results = [None] * len(templates)
      for i, template in enumerate(templates):
        results[i] = match_template(screen, screen_image, template)
      results = sum(results, [])

    for (card_data, (max_point_x, max_point_y, score), color) in results:
      max_point = (max_point_x, max_point_y)
      print(card_data['name'], max_point, score, color)
      draw_rectangle(screen, max_point, choose_color(color))

    cv2.imshow('my_screen', screen)
    end_time = time.time() - start_time
    print(f"Time Elapsed: {end_time}")
    cv2.waitKey(0)
    print("Received keystroke")
    key = keyboard.read_key()
    print(f"Evaluating key '{key}'")
    if key == 'esc':
      should_continue = False
    else:
      continue

  cv2.destroyAllWindows()

def main():
  if PARALLEL:
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
    try:
      run_program(pool)
    finally:
      pool.close()
  else:
    run_program()

if(__name__ == "__main__"):
  # Create a cProfile object
  profiler = cProfile.Profile()

  # Start the profiler
  profiler.enable()

  # Run the code you want to profile
  main()

  # Stop the profiler
  profiler.disable()

  # Print the profiling results
  profiler.print_stats(sort='tottime')
  profiler.dump_stats("profile.txt")
