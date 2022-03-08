import imgaug as ia
from imgaug import augmenters as iaa
from pascal_voc_writer import Writer
from os import listdir
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

#time library
from datetime import datetime

ia.seed(1)

def read_anntation(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bounding_box_list = []

    file_name = root.find('filename').text

    for obj in root.iter('object'):

        object_label = obj.find("name").text
        for box in obj.findall("bndbox"):
            x_min = int(box.find("xmin").text)
            y_min = int(box.find("ymin").text)
            x_max = int(box.find("xmax").text)
            y_max = int(box.find("ymax").text)

        bounding_box = [object_label, x_min, y_min, x_max, y_max]
        bounding_box_list.append(bounding_box)

    return bounding_box_list, file_name

def read_train_dataset(dir):
    images = []
    annotations = []

    for file in listdir(dir):
        if '.jpg' in file.lower() or '.png' in file.lower():
            images.append(cv2.imread(dir + file, 1))
            annotation_file = file.replace(file.split('.')[-1], 'xml')
            bounding_box_list, file_name = read_anntation(dir + annotation_file)
            annotations.append((bounding_box_list, annotation_file, file_name))

    images = np.array(images)

    return images, annotations


 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 
def start_aug(image_folder, augSeq):

    abc=1
    current_time=datetime.now()
    current_time=str(current_time.year)+'_'+str(current_time.month)+'_'+str(current_time.day)+'_'+str(current_time.hour)+'_'+str(current_time.minute)

    createFolder(image_folder+current_time)


    for i in range(abc):
    #횟수 조절 하는 코드

        dir = image_folder
        dir2 = dir+current_time+'/'

        images, annotations = read_train_dataset(dir)
        print(len(images),"\n")

        for idx in range(len(images)):
            image = images[idx]
            boxes = annotations[idx][0]

            ia_bounding_boxes = []
            for box in boxes:
                ia_bounding_boxes.append(ia.BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4]))

            bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)

            seq = iaa.Sequential(augSeq)

            seq_det = seq.to_deterministic()

            image_aug = seq_det.augment_images([image])[0]
            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

            
            new_image_file = dir2 + current_time + annotations[idx][2]
            #print(new_image_file)
            if '.' in new_image_file:
                new_image_file= new_image_file.replace('.jpg',"",1)
                print(new_image_file)

            cv2.imwrite(new_image_file, image_aug)

            h, w = np.shape(image_aug)[0:2]
            voc_writer = Writer(new_image_file, w, h)

            for i in range(len(bbs_aug.bounding_boxes)):
                bb_box = bbs_aug.bounding_boxes[i]
                voc_writer.addObject(boxes[i][0], int(bb_box.x1), int(bb_box.y1), int(bb_box.x2), int(bb_box.y2))

            voc_writer.save(dir2 + current_time + annotations[idx][1])
            print(idx/len(images)*100)

    print("complete! ")
