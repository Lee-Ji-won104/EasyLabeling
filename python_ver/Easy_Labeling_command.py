# -*- coding: utf-8 -*-
from tqdm import tqdm
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from os import listdir
from xml.etree.ElementTree import Element, SubElement, ElementTree
#from tflite_model_maker.config import ExportFormat
#from tflite_model_maker import model_spec
#from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


def read_train_dataset(dir):
    images = []

    for file in listdir(dir):
        if 'jpg' in file.lower() or 'png' in file.lower():
            images.append(cv2.imread(dir + file, 1))

    images = np.array(images)

    return images

def preprocess_image(image_path, input_size):
  """Preprocess the input image to feed to the TFLite model"""
  img = tf.io.read_file(image_path)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.uint8)
  original_image = img
  resized_img = tf.image.resize(img, input_size)
  resized_img = resized_img[tf.newaxis, :]
  return resized_img, original_image


def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold, model_type):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  if model_type=="speed":
    #300 ssd model
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

  elif model_type=="accurate":
    #efficient model
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))

  else:
    #efficient model
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))
  

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_odt_and_draw_results(image_path, imageName, interpreter, threshold, model_type):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path, 
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold, model_type=model_type)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)

  makeAnnotation(imageName, results, original_image_np)
    
  #print("success")


def makeAnnotation(imageName, results, original_image_np ):
    filename = imageName #이름을 넣어주자
    width = original_image_np.shape[1]
    height = original_image_np.shape[0]
 
    root = Element('annotation')
    SubElement(root, 'folder').text = 'ijiwon'
    SubElement(root, 'filename').text = filename + '.jpg'  #jpg로 할 지 다른 것으로 할지는 나중에 인풋으로 넣어보자
    SubElement(root, 'path').text = './object_detection/images' +  filename + '.jpg' #jpg로 할 지 다른 것으로 할지는 나중에 인풋으로 넣어보자
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'
 
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = '3'
 
    SubElement(root, 'segmented').text = '0'

    ###################나중에 위치 옮기자
    #classes=['child_sign','person','car_rear','bicycle','kickboard','motorcycle','30','40','50','60','70']
    
    #편의점 용 classes
    #classes=['beer_berni', 'beer_blanc', 'beer_bud', 'beer_cass_fresh', 'beer_cloud_original', 'beer_filgood_7', 'beer_filgood_original', 'beer_filite_original', 'beer_heineken', 'beer_terra', 'beer_tsingtao', 'can_2per', 'can_chilsung', 'can_coca_zero', 'can_demi_apple', 'can_demi_lemon', 'can_fanta_orange', 'can_gatorade', 'can_mac_col', 'can_milkis', 'can_mountdew', 'can_pearjuice', 'can_pocari', 'can_powerade', 'can_sprite', 'can_toreta', 'juice_capri-sun_orange', 'milk_banana', 'milk_banana_lite', 'milk_chocochoco', 'milk_coffeecoffee', 'milk_deliciousmilk_300', 'soju_drop_fresh', 'soju_drop_jamong', 'soju_drop_original', 'soju_goodday_blueberry', 'soju_jinro', 'soju_likefirst', 'soju_likefirst_soft', 'soju_maehwa', 'viyott_chococrispy', 'viyott_cookiecream', 'viyott_crunch']
    
    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute 
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * original_image_np.shape[1])
        xmax = int(xmax * original_image_np.shape[1])
        ymin = int(ymin * original_image_np.shape[0])
        ymax = int(ymax * original_image_np.shape[0])
        # Find the class index of the current object
        class_id = classes[int(obj['class_id'])]

        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = class_id
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(xmin)
        SubElement(bbox, 'ymin').text = str(ymin)
        SubElement(bbox, 'xmax').text = str(xmax)
        SubElement(bbox, 'ymax').text = str(ymax)
    
    fileName=imageName.rsplit('.')[0]
    tree = ElementTree(root)
    tree.write(INPUT_IMAGE_URL+fileName+'.xml')

def read_labels(label_txt):
  file=open(label_txt,"r")
  while True:
    line=file.readline()
    if not line:
      break
    classes.append(line.strip())
  
  file.close()


if __name__ == "__main__":

  #default
  model_type="accurate"
  DETECTION_THRESHOLD = 0.5
  INPUT_IMAGE_URL = './images/'
  model_path='./models/efficientdet4.tflite'
  label_txt='./models/labels.txt'
  classes=[]

  #read label
  label_txt=str(input("please type the location of labels.txt: "))
  if not label_txt:
    print("wrong location -> default labels.txt")
  read_labels(label_txt)

  #인풋 모델
  model_path=str(input("please type the location of TFlite model: "))
  if not model_path:
    model_type=str(input("you are going to use default model. please type what type of model you want( speed / accurate )"))
  
  #인풋 사진
  INPUT_IMAGE_URL=str(input("please type the directory of images: "))

  #컨피던스 조절
  DETECTION_THRESHOLD=float(input("please type the threshhold(0~1 default 0.4):n "))
  if DETECTION_THRESHOLD>1 or DETECTION_THRESHOLD<0:
    print("you typed wrong number...")

  print("DETECTION_THRESHOLD = "+DETECTION_THRESHOLD)

  # Load the TFLite model
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  imagesTo=[]

  for file in listdir(INPUT_IMAGE_URL):
      imagesTo.append(file)

  imagesTo.sort()

  pbar=tqdm(total=len(imagesTo))
  i=0

  for file in imagesTo:
      pbar.update(i)
      i+=1
      detection_result_image = run_odt_and_draw_results(
              INPUT_IMAGE_URL+file, 
              file,
              interpreter, 
              threshold=DETECTION_THRESHOLD,
              model_type=model_type
          )
  pbar.close()

