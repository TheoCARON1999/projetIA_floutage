from ultralytics import YOLO
import os
import cv2
from IPython.display import display,Image

import subprocess
import requests
import json
import random
import base64
import torch

from roboflow import Roboflow

#Lien dataset roboflow : https://app.roboflow.com/work-ewkyd/foruni/2
#Création du dataset
rf = Roboflow(api_key="PSppi2E4iOFo29ZqM3qp")
project = rf.workspace("work-ewkyd").project("foruni")
dataset = project.version(2).download("yolov8")

"""
model = YOLO('yolov8n.pt') 
model.to('cuda')

# Train the model
# necessite de modifier différents paths dans les .yaml du dataset pour fonctionner"
results = model.train(data='./ForUni-2/data.yaml', epochs=150)
"""

"""
#Predict
# Load a model
#model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('./pathtoBestModel.pt')  # load best.pt created by the training
"""

"""
# Predict with the model
results = model.predict('./car.jpg',show=True)  # predict on an image
"""