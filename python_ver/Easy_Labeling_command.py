# -*- coding: utf-8 -*-
#from tqdm import tqdm
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from os import listdir
from xml.etree.ElementTree import Element, SubElement, ElementTree

import tensorflow as tf
assert tf.__version__.startswith('2')

#time module
import time

#broke_image_check
import libs.check_image

#check_precision
import libs.check_precision

#augmenatations
import libs.imgaug_object_detection

from imgaug import augmenters as iaa

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


def detect_objects(interpreter, image, threshold):
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
    #tensorflow model maker
    boxes = get_output_tensor(interpreter, 1)
    classes = get_output_tensor(interpreter, 3)
    scores = get_output_tensor(interpreter, 0)
    count = int(get_output_tensor(interpreter, 2))

  else:
    #efficient model
    #this is default now.
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))
  

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


def run_odt_and_draw_results(image_path, imageName, interpreter, threshold):
  """Run object detection on the input image and draw the detection results"""
  # Load the input shape required by the model
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(
      image_path, 
      (input_height, input_width)
    )

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)

  drawResult=makeAnnotation(imageName, results, original_image_np)
  if showImage==True:
    return drawResult
  #print("success")

def makeAnnotation(imageName, results, original_image_np ):
    filename = imageName #이름을 넣어주자
    fileName2=imageName.rsplit('.')[0]

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

    for obj in results:
        # Convert the object bounding box from relative coordinates to absolute 
        # coordinates based on the original image resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        
        xmin = int(xmin * original_image_np.shape[1])
        if xmin<0:
          xmin=0

        xmax = int(xmax * original_image_np.shape[1])
        if xmax>width:
          xmax=width

        ymin = int(ymin * original_image_np.shape[0])
        if ymin<0:
          ymin=0

        ymax = int(ymax * original_image_np.shape[0])
        if ymax>height:
          ymax=height

        #This is Option
        if showImage==True:
          red_color=(0,0,255)
          drawImage=cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax),red_color ,2)

        # Find the class index of the current object
        class_id = classes[int(obj['class_id'])]

        #Check Precision
        if checkPrecision== True:
          if class_id != None:
            libs.check_precision.calculate_MBTI(class_id, fileName2)
          
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

    #This is Option
    if showImage==True:
      return drawImage


def read_labels(label_txt):
  file=open(label_txt,"r")
  while True:
    line=file.readline()
    if not line:
      break
    classes.append(line.strip())
  
  file.close()

  #...
  print(classes)


if __name__ == "__main__":

  #default
  model_type="accurate"
  DETECTION_THRESHOLD = 0.4
  INPUT_IMAGE_URL = './images/'
  model_path='./models/efficientdet4.tflite'
  label_txt='./models/labels.txt'
  classes=[]
  showImage=False

  #check showing images
  showImg=str(input('Do you want to check the bounding boxes of photos? (Y/N) '))
  if showImg=='Y' or showImg=='y':
    showImage=True
    print('you can check the bounding boxes')

  elif showImg=='N' or showImg=='n':
    showImage=False

  else:
    showImage=False

  #check precision of model
  check_P=str(input('Are you going to check the precision of your model? (Y/N)' ))
  if check_P=='Y' or check_P=='y':
    checkPrecision=True
    print(' This machine will check the precision of your model ')
  elif check_P=='N' or check_P=='n':
    checkPrecision=False
    print('THis machine won\'t check the precision of your model')
  else:
    print('Wrong Input... this machine won\'t check the precision of your model')
    check_P=False

  #read label
  label_txt=str(input("please type the location of labels.txt: "))
  if not label_txt:
    print("wrong location -> default labels.txt")
    label_txt='./models/labels.txt'
  print(label_txt)
  read_labels(label_txt)

  #인풋 모델
  model_path=str(input("please type the location of TFlite model: "))
  if not model_path:
    model_type=str(input("you are going to use default model. please type what type of model you want( speed / accurate )"))
    if model_type=='speed':
      model_path='./models/ssd_mobilenet.tflite'
    elif model_type=='accurate':
      model_path='./models/efficientdet4.tflite'
    else:
      model_path='./models/efficientdet4.tflite'
  print(model_path)

  #인풋 사진
  INPUT_IMAGE_URL=str(input("please type the directory of images: "))
  if not INPUT_IMAGE_URL:
    INPUT_IMAGE_URL = './images/'
    
  print(INPUT_IMAGE_URL)

  #check image is good or bad.
  #if images's condition is bad, 
  #remove that images.
  libs.check_image.start_check(INPUT_IMAGE_URL)

    
  #컨피던스 조절
  DETECTION_THRESHOLD=float(input("please type the threshhold(0~1 default 0.4): "))
  if DETECTION_THRESHOLD>1 or DETECTION_THRESHOLD<0:
    print("you typed wrong number...")
    DETECTION_THRESHOLD = 0.4

  print("DETECTION_THRESHOLD = "+str(DETECTION_THRESHOLD))

  # Load the TFLite model
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  imagesTo=[]

  for file in listdir(INPUT_IMAGE_URL):
      imagesTo.append(file)

  imagesTo.sort()

  #pbar=tqdm(total=len(imagesTo))
  i=0

  for file in imagesTo:
    
    #pbar.update(i)
    #i+=1
    print('-------------------------------')
    print(file+"  is Easy-Labeling!!")
    print(str(i/len(imagesTo)*100)+' is done.')
    i+=1
    firstTime= time.time()
  
    detection_result_image = run_odt_and_draw_results(
            INPUT_IMAGE_URL+file, 
            file,
            interpreter, 
            threshold=DETECTION_THRESHOLD,
        )
    if showImage==True and detection_result_image.all():
      cv2.imshow('result_image',detection_result_image)
  
    secondTime=time.time()
    secondTime-=firstTime
    print("Inference Time: "+str(secondTime))
    print('-------------------------------')
  #pbar.close()

  if checkPrecision==True:
    result_list=libs.check_precision.calculate_All()
    d=['accuracy','recall','precision','F1']
    ak47=0

    for a in result_list:
      print(d[ak47]+' is ',end='')
      print(str(a))
      ak47+=1

  while True:
      
    # Augmenatation
    augQuest=str(input("Do you want to Augmentate your Datasets? (Y/N): "))
    augSeq=[]

    if augQuest=="Y":
      
      print("What type of augmentations do you want?:")

      while True:
        print("******fog*****")
        print("****affine****")
        print("*gausian blur*")
        print("*****rain*****")
        print("***snowflake**")
        print("     exit      ")
        
        augType=str(input("please choose: "))
        if augType=="fog":
          augSeq.append(iaa.Fog())
        
        elif augType=="affine":
          augSeq.append(iaa.Affine(scale=(0.3, 0.7)))

        elif augType=="gausian blur":
          augSeq.append(iaa.GaussianBlur(sigma=(0.5, 3.0)))
        
        elif augType=="rain":
          augSeq.append(iaa.Rain(speed=(0.1, 0.3)))

        elif augType=="snowflake":
          augSeq.append(iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05)))

        elif augType=="exit":
          break

        else:
          print("you typed the wrong words.")


      libs.imgaug_object_detection.start_aug(INPUT_IMAGE_URL,augSeq=augSeq)
      break

    elif augQuest=="N":

      print("Ok, All process is complete!!!!!! Enjoy your Deep Learning!")
      quit()

    else:
      print(" You Typed Wrong. Please Type Y/N.")

  cv2.destroyAllWindows()

