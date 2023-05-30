import io
import shutil
import webbrowser
import requests
import os
import json
from PIL import Image

IMAGE_PATH = "images\png"

def get_card_list():
  url = "https://triad.raelys.com/api/cards"

  json_response = json.loads(requests.get(url).content)

  with open("cards.json", 'w', encoding='utf-8') as f:
    json.dump(json_response, f, ensure_ascii=False, indent=4)   

def read_card_list():
  try:
    with open("cards.json", 'r', encoding='utf-8') as f:
      json_object = json.load(f)['results']
    return json_object
  except Exception as ex:
    print(f"Error when reading card json file: {ex.message}")

def get_card_filename(card):
  return os.path.join(IMAGE_PATH, f"card_{card['id']}.png")

def get_card_image(card):
  url = card['image']
  file_location = get_card_filename(card)
  image_response = requests.get(url)
  in_memory_file = io.BytesIO(image_response.content)
  im = Image.open(in_memory_file)
  im.save(file_location)

def load_all_data():
  get_card_list()
  card_json = read_card_list()
  for card in card_json:
    get_card_image(card)

if(__name__ == "__main__"):
  load_all_data()

