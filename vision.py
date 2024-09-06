#-----------Import Commands-----------
import io
import os.path
import requests
import json
from pprint import pprint
# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
class Vision:

    def __init__(self):
        print("Initializing Vision Class")

    def process_image(self, frame):

        client = vision.ImageAnnotatorClient()

        # The name of the image file to annotate


        # Loads the image into memory
        with io.open(frame, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Performs label detection on the image file
        response = client.label_detection(image=image)
        labels = response.label_annotations
        label_list = []
        for label in labels:
            label_list.append(str(label.description).lower())
        self.check_label(label_list)
        print(label_list)

    def check_label(self, label_list):
        for label in label_list:
            if (str(label.lower()) == 'algae'):
                print("Algae detected!")
