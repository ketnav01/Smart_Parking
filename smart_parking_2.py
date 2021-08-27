# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:55:14 2021

@author: Ketnav
"""

from os import listdir
from xml.etree import ElementTree as ET
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.model import load_image_gt
from mrcnn.utils import compute_ap
from mrcnn import model as modellib
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import os
import numpy as np


class ParkingLot(Dataset):
	
    """
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class('parking', 1, 'ParkingOccupied')
        self.add_class('parking', 2, 'ParkingEmpty')
        if is_train:
            # Path
            dataset_path = '/parkinglot/'
            #dataset_path = "parkinglot"
            img_dir = os.path.join(dataset_path, 'train/images')
            labels_dir = os.path.join(dataset_path, 'train/labels')
        elif not is_train:
            img_dir = os.path.join(dataset_path, 'test/images')
            labels_dir = os.path.join(dataset_path, 'test/iabels')
        for filename in os.listdir(img_dir):
            image_id = filename[:-4]
            img_path = os.path.join(img_dir, filename)
            label_path = labels_dir + '/' + image_id + '.xml'
            self.add_image('parking', image_id=image_id, path=img_path, annotation=label_path)
     """
    
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class('parking', 1, 'ParkingOccupied')
        self.add_class('parking', 2, 'ParkingEmpty')
        if is_train:
            img_dir = dataset_dir + '/train/images/'
            labels_dir = dataset_dir + '/train/labels'
        elif not is_train:
            img_dir = dataset_dir + '/test/images/'
            labels_dir = dataset_dir + '/test/labels'
        for filename in listdir(img_dir):
            image_id = filename[:-4]
            img_path = img_dir + filename
            label_path = labels_dir + '/' + image_id + '.xml'
            self.add_image('parking', image_id=image_id, path=img_path, annotation=label_path)


    def extract_contours(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        contours = []
        for space in root.getiterator('space'):
            cls_id = int(space.attrib['occupied'])
            for rect in space.findall('rotatedRect/center'):
                x = int(rect.get('x'))
                y = int(rect.get('y'))
            for rect in space.findall('rotatedRect/size'):
                w = int(rect.get('w'))
                h = int(rect.get('h'))
            for rect in space.findall('rotatedRect/angle'):
                d = int(rect.get('d'))
            coors = [cls_id, x, y, w, h, d]
            contours.append(coors)
        return contours


    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        contours = self.extract_contours(path)
        mask = np.zeros([720, 1280, len(contours)], dtype='uint8')
        class_ids = []
        for i in range(len(contours)):
            contour = contours[i]
            cls_id, x, y, w, h, d = contour
            row_s, row_e = y-(w), y+(w//2)
            col_s, col_e = x-(h//2), x+(h//2)
            mask[row_s:row_e, col_s:col_e, i] = 1
            if cls_id==1:
                class_ids.append(self.class_names.index('ParkingOccupied'))
            elif cls_id==0:
                class_ids.append(self.class_names.index('ParkingEmpty'))
		
        return mask, np.asarray(class_ids, dtype='int32')
		
    def image_ref(self, image_id):
        info = self.image_info[image_id]
        return info['path']

"""
class ParkingConfig(Config):
    NAME = 'parking_cfg'
    NUM_CLASSES = 1 + 2
    STEPS_PER_EPOCH = 50


# train set
train_set = ParkingLot()
#train_set.load_dataset('ParkingOccupied', is_train=True)
train_set.load_dataset('parkinglot', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))


# test/val set
test_set = ParkingLot()
#test_set.load_dataset('ParkingOccupied', is_train=False)
test_set.load_dataset('parkinglot', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


# enumerate all images in the dataset to verify if images and xml loaded correctly
for image_id in train_set.image_ids:
	# load image info
	info = train_set.image_info[image_id]
	# display on the console
	print(info)



# prepare config
config = ParkingConfig()
config.display()
model = MaskRCNN(mode='training', model_dir='./', config=config)
#model = modellib.MaskRCNN(mode='training', model_dir='./', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_mask'])
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=2, layers='heads')

"""

# Evaluate a Mask R-CNN Model

class PredictionConfig(Config):
    NAME = 'parking_cfg'
    NUM_CLASSES = 1 + 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# train set
train_set = ParkingLot()
train_set.load_dataset('parkinglot', is_train=True)
train_set.prepare()
print(f'Train: {len(train_set.image_ids)}')
test_set = ParkingLot()
test_set.load_dataset('parkinglot', is_train=False)
test_set.prepare()
print(f'Test:{len(test_set.image_ids)}')
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('mask_rcnn_parking_cfg_0002.h5', by_name=True)

# Draw Image with dtected objects

def draw_image_with_boxes(filename, boxes_list, class_list):
	data = plt.imread(filename)
	plt.figure(figsize=(20, 14))
	plt.imshow(data)
	# get the context for drawing boxes
	ax = plt.gca()
	# plot each box
	for box, cls in zip(boxes_list, class_list):
		# get coordinates
		y1, x1, y2, x2 = box
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		if cls==1:
			rect = Rectangle((x1, y1), width, height, fill=False, color='blue')
		elif cls==2:
			rect = Rectangle((x1, y1), width, height, fill=False, color='green')
		#draw the box
		ax.add_patch(rect)
	
	# show the plot
	plt.show()

"""
# Predictions 1

# load any random image, detect the available and occupied parking spots, use our function to draw them

img = load_img('./parkinglot/test/images/2013-03-20_12_50_07.jpg')
#img = load_img('./parkinglot/test/images/2013-03-20_12_50_07.jpg')
img = img_to_array(img)
# make prediction
results = model.detect([img], verbose=0)
# visualize the results

# draw_image_with_boxes('./parkinglot/test/images/2013-03-20_12_50_07.jpg', results [0]['rois'], results[0]['class_ids'])


# Visualize the result by detecting the results by drawing region of interests (ROIs)
draw_image_with_boxes('./parkinglot/test/images/2013-03-20_12_50_07.jpg', results [0]['rois'],\
results[0]['class_ids'])

    
# Drawing the mask - Mask R-CNN visualization Method

class_names = ['BG', 'ParkingOccupied', 'ParkingEmpty']
img = load_img('./parkinglot/test/images/2013-03-20_12_50_07.jpg')
img = img_to_array(img)
# make prediction
results = model.detect([img], verbose=0)
# get dictionary for first prediction
r = results[0]
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

"""

"""
# Predictions 2

# load any random image, detect the available and occupied parking spots, use our function to draw them

img = load_img('2012-10-27_17_46_10.jpg')
#img = load_img('./parkinglot/test/images/2013-03-20_12_50_07.jpg')
img = img_to_array(img)
# make prediction
results = model.detect([img], verbose=0)
# visualize the results

# draw_image_with_boxes('./parkinglot/test/images/2013-03-20_12_50_07.jpg', results [0]['rois'], results[0]['class_ids'])


# Visualize the result by detecting the results by drawing region of interests (ROIs)
draw_image_with_boxes('2012-10-27_17_46_10.jpg', results [0]['rois'],\
results[0]['class_ids'])
"""
    
# Drawing the mask - Mask R-CNN visualization Method

class_names = ['BG', 'ParkingOccupied', 'ParkingEmpty']
img = load_img('2012-10-27_17_46_10.jpg')
img = img_to_array(img)
# make prediction
results = model.detect([img], verbose=0)
# get dictionary for first prediction
r = results[0]
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

