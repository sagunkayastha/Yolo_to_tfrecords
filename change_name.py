from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import shutil

from y2p.pascal_voc_io import XML_EXT
from y2p.pascal_voc_io import PascalVocWriter
from y2p.yolo_io import YoloReader
import os.path
import sys

import glob
import pandas as pd
import xml.etree.ElementTree as ET


try:
    
    from PyQt5.QtGui import QImage
   
except ImportError:
    from PyQt4.QtGui import QImage


###



import io
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

from y2p.tfutils import split,create_tf_example


def change_name(files):
    for i in sorted(files):
        shutil.move('obj/'+i,'obj/'+i.replace('-',''))

def mkdirs():
    try:
        shutil.rmtree('Data')
    except:
        pass
    os.makedirs('Data/images', exist_ok=True)
    os.makedirs('Data/txt', exist_ok=True)
    os.makedirs('Data/labels', exist_ok=True)

def separate_img_xml(imgFolderName,imgFileName):
    file_ =os.path.join(imgFolderName,imgFileName)
    if '.png' in file_:
        shutil.copy(file_,os.path.join('Data/images/'+imgFileName))
    if '.txt' in file_:
        shutil.copy(file_,os.path.join('Data/txt/'+imgFileName))
    

def yolo_to_pascal(imgFolderPath):

    # Search all yolo annotation (txt files) in this folder
    for file in os.listdir(imgFolderPath):
        separate_img_xml(imgFolderPath,file)
        
        if file.endswith(".txt") and file != "classes.txt":
            print("Convert", file)

            annotation_no_txt = os.path.splitext(file)[0]

            imagePath = imgFolderPath + "/" + annotation_no_txt + ".png"

            image = QImage()
            image.load(imagePath)
            imageShape = [image.height(), image.width(), 1 if image.isGrayscale() else 3]
            imgFolderName = os.path.basename(imgFolderPath)
            imgFileName = os.path.basename(imagePath)
            
            


            writer = PascalVocWriter(imgFolderName, imgFileName, imageShape, localImgPath=imagePath)


            # Read YOLO file
            txtPath = imgFolderPath + "/" + file
            tYoloParseReader = YoloReader(txtPath, image,  classListPath='classes.txt')
            shapes = tYoloParseReader.getShapes()
            num_of_box = len(shapes)

            for i in range(num_of_box):
                label = shapes[i][0]
                xmin = shapes[i][1][0][0]
                ymin = shapes[i][1][0][1]
                x_max = shapes[i][1][2][0]
                y_max = shapes[i][1][2][1]
            
                
                
                writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

            writer.save(targetFile= 'Data/labels' + "/" + annotation_no_txt + ".xml")

def xml_to_csv(path):
    xml_list = []
    imgdir=os.path.join('Data','images')+'/'
    for xml_file in glob.glob(path+'/*.xml'):
        print(xml_file)
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (imgdir+root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def class_text_to_int(row_label):
	dict = {'background':0,'POL':1 ,'ANOM':2 , 'MOL':3, 'OTHPAR':4}
	return dict[row_label]	
    

def pascal_to_tf(csv_input,output_path):
    


    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(os.getcwd(), 'images')
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


def main():
    mkdirs()
    folder_name = 'obj'
    files = os.listdir(folder_name)
    label_file='label.csv'
    
    output_path = 'tfrecords/train.tfrecords'


    # #Step 1
    # change_name(files)
    
    # # NEED classes.txt in the folder
    # imgFolderPath = folder_name
    # yolo_to_pascal(imgFolderPath)

    # ## make label.csv

    # xml_df = xml_to_csv('Data/labels')
    # xml_df.to_csv(('labels.csv'), index=None)

    pascal_to_tf(label_file,output_path)


main()

