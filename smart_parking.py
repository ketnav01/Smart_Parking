# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 14:50:52 2021

@author: Ketnav
"""

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.model import load_image_gt
from mrcnn.utils import compute_ap


class ParkingLot(Dataset):
	
	def load_dataset (self, dataset_dir, is_train=True):
		self.add_class('parking', 1, 'ParkingOccupied')
		self.add_class('parking', 2, 'ParkingEmpty')
		if is_train:
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
		
	def image_ref (self, image_id):
		info = self.image_info[image_id]
		return info['path']