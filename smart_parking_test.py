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
				class_ids.append(self.class_names. index('ParkingEmpty'))
		
		return mask, np.asarray(class_ids, dtype='int32')
		
	def image_ref (self, image_id):
		info = self.image_info[image_id]
		return info['path']



class ParkingConfig(Config):
	NAME = 'parking_cfg'
	NUM_CLASSES = 1 + 2
	STEPS_PER_EPOCH = 50

# train set
train_set = Parking Lot()
train_set.load_dataset('ParkingOccupied', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
test_set = Parkinglot()
test_set.load_dataset('ParkingOccupied', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
config = ParkingConfig()
config.display()
model = modellib.MaskRCNN(mode='training', model_dir=model_dir, config=config)
model.load_weights(coco_model_path, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_mask'])
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')



class PredictionConfig(Config):
NAME = 'parking_cfg'
NUM_CLASSES = 1 + 2
GPU_COUNT = 1
IMAGES_PER_GPU = 1



# train set
train_set = Parking Lot()
train_set.load_dataset('ParkingOccupied', is_train=True)
train_set.prepare()
print(f'Train: {len(train_set.image_ids)}')
test_set = ParkingLot()
test_set.load_dataset('ParkingOccupied', is_train=False)
test_set.prepare()
print(f'Test:{len(test_set.image_ids)}')
cfg = PredictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights('mask_rcnn_parking_cfg_0005.h5', by_name=True)



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
			rect = patches.Rectangle((x1, y1), width, height, fill=False, color='blue')
		elif cls==2:
			rect = patches.Rectangle((x1, y1), width, height, fill=False, color='green')
		#draw the box
		ax.add_patch(rect)
	
	# show the plot
	plt.show()



img = load_img('./parkinglot/test/images/2013-03-20_12_50_07.jpg')
imo = img_to_array(img)
# make prediction
results = model.detect([img], verbose=0)
# visualize the results
draw_image_with_boxes('./parkinglot/test/images/2013-03-20_12_50_07.jpg', results [0]['rois'],\
results[0]['class_ids'])



class_names = ['BG', 'ParkingOccupied', 'ParkingEmpty')
img = load_img('./parkinglot/test/images/2013-03-20_12_50_07.jpg')
img = img_to_array(img)
# make prediction
results = model.detect([img], verbose=0)
# get dictionary for first prediction
r = results[0]
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])


img = load_img('/Users/angela/Downloads/parking_test2.jpg')
img = img_to_array(img)
# make prediction
results = model.detect([img], verbose=0)
# visualize the results
draw_image_with_boxes('/Users/angela/Downloads/parking_test2.jpg', results[0] ['rois'], results[0] ['class_ids'])


