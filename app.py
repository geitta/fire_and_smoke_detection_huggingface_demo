import gradio as gr
import os
import torch
from timeit import default_timer as Timer
from ultralytics import YOLO

# model preparation
model = YOLO("best.pt")

# predict function
def predict(img):
  """performs prediction on an image and returns prediction and time taken"""
  # start the timer
  start_time = timer()
  # convert image to an array
  img_np = np.array(img)
  # make prediction on image
  predictions = model.to(device).predict(source=img_np, conf=0.5)
  # get the first results object
  prediction = predictions[0]
  # render bounding boxes
  annotated = prediction.plot()
  # numpy to pil
  pred = Image.fromarray(annotated)
  # calculate the prediction time
  pred_time = round(timer() - start_time, 5)
  # return the prediction and prediction time
  return pred, pred_time

# gradio app
# create title, description and article strings
title = "fire and smoke detection"
description = "object detection model trained on the YOLOv8 architecture"
article = "created at [repo](https://colab.research.google.com/drive/1AyOw98JSKtm6iQuIArmqloH2WwaRLIn8#scrollTo=2wtGeuPg_tpo)"

# create the gradio demo
demo = gr.Interface(fn=predict, #mapping function: from input to output
                   inputs = gr.Image(type="pil"), # what are the inputs
                   outputs=[gr.Image(type='pil', label="prediction"), # what are our outputs, our function has two outputs therefore we have two outputs
                            gr.Number(label="prediction time (s)")],
                   examples=example_list,
                   title=title,
                   description=description,
                   article=article
                   )
# launch the demo
demo.launch()
