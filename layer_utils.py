
import numpy as np
import csv
import config
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from random import randint


def anchor_boxes(feature_shape,
				image_shape,
				index= 0 ,
				n_layers =4,
				aspect_ratios = (1,2,0.5)):
	'''
	Compute the anchor boxes for a given feature map
	Anchor boxes are in minmax format
	Arguments:
		feature_shape(list): Feature map shape
		image_shape(list):   Image size shape
		index(int):          Indicates which of ssd head layers are we referrung to
		n_layers(int):       Achor boxes per feature map
	'''
	# anchor box  sizes  given an index of layer in ssd head 
	size = anchor_boxes(n_layers)[index]

	# number of anchor boxes per feature map pt 
	n_boxes = len(aspect_ratios)+1

	# ignore number of channels(last)
	image_height,image_width ,_ = image_shape

	#ignore number of feature maps(last)
	feature_height,feature_width,_ = feature_shape

	# normalized width and height
	# size[0] is scale ,size[1] is sqrt(scale*(sclae+1))
	norm_height = image_height*size[0]
	norm_width = image_width*size[0]

	# list of anchor boxes(width,height)
	width_height = []
	# anchor box by aspect ration on resized image dims 
	# equation 11.2.3
	for ar in aspect_ratios:
		box_width = norm_width*np.sqrt(ar)
		box_height = norm_height/np.sqrt(ar)

		width_height.append((box_width,box_height))

	# multipy anchor box dim by size[1] for aspect_ratio = 1
	# equation 11.2.4
	box_width = image_width*size[1]
	box_height = image_height*size[1]
	width_height.append((box_width,box_height))

	# now an array of (width,height)
	width_height = np.arrau(width_height)

	# dimensions of each receptive field in pixels
	grid_width = image_width/feature_width
	grid_height = image_height/feature_shape

	# compute center of receptive field featurs pt
	# (cx,cy) format

	# starting at midpoint of lst receptive field
	start = grid_width*0.5
	# ending at midpoint of last receptive field
	end = (feature_width-0.5)*grid_width

	cx = np.linespace(start,end,feature_width)
	start = grid_height*0.5
	end = (feature_height-0.5)*grid_height
	cy = np.linespace(start,end,feature_height)

	# grid of box centers
	cx_grid,cy_grid = np.meshgrid(cv,cy)

	# for np.tile()
	cx_grid = np.expand_dims(cx_grid,-1)
	cy_grid = np.expand_dims(cy_grid,-1)


	# tensor = (feature_map_height,feature_map_width,n_boxes,4)
	# aligned with image tesnsor(height,width,channels)
	# lasr dimension = (cx,cy,w,h)
	boxes = np.zeros((feature_height,feature_width,n_boxes,4))

	# (cx,cy)
	boxes[...,0] = np.tile(cx_grid,(1,1,n_boxes))
	boxes[...,1] = np.tile(cy_grid,(1,1,n_boxes))

	# (w,h)
	boxes[...,2] = width_height[:,0]
	boxes[...,3] = width_height[:,1]

	# convert (cx,cy,w,h) to (xmin,xmax,ymin,ymax)
	# prepend one dimemsion to boxes 
	# to account for the batch size = 1 
	boxes = centroid2minmax(boxes)
	boxes = np.expand_dims(boxes,axis=0)
	return boxes 


def centroid2minmax(boxes):
	'''
	Centroid to minmax format
	(cx,cy,w,h) to (xmin,xmax,ymin,ymax)
	Arguments:
		boxes(tensor):	Batch of boxes in centroid format
	Returns:
		minmax(tensor): Batch of boxes in minmax format
	'''
	minmax = np.copy(boxes).astype(np.float)
	minmax[...,0] = boxes[...,0] - (0.5*boxes[...,2])

	minmax[...,1] = boxes[...,0] + (0.5*boxes[...,2])
	
	minmax[...,2] = boxes[...,1] - (0.5*boxes[...,3])
	
	minmax[...,3] = boxes[...,1] + (0.5*boxes[...,3])
	return minmax



def get_gt_data(iou,n_classes = 4,anchors = None,labels = None,normalize = False,threshold=0.6):
	'''
	Retrieve ground truth class bbox offset,and mask 
	Arguments:
		iou(tensor):	 IoU of each bounding box wrt each anchor box 
		n_classes(int):  Number of object classes
		anchors(tensor): Anchor boxes per feature layer
		labels(list):    Ground truth labels 
		normalize(bool): If normalization should be applied
		threshold(float):If less than 1.0 ,anchor boxes > threshold are also part of postive anchor boxes
	Returns:
		gt_class,gt_offset,gt_mask (tensor): Ground truth classess,offsets and masks
	'''
	# each maxiou_per_get is index of anchor w/max iou 
	# for the given ground truth bouding box
	"""Equation 11.3.2""" 
	maxiou_per_gt = np.argmax(iou,axis=0)

	# get extra anchor boxes based on IoU
	if threshold<1.0:
		"""Extra positive anchor boxes are detemined based on a user-defined threshold"""
		iou_gt_threshold = np.argwhere(iou>threshold)
		if iou_gt_threshold.size > 0:
			extra_anchors = iou_gt_threshold[:,0]
			extra_classes = iou_gt_threshold[:,1]
			extra_labels = labels[extra_classes]

			indexes = [maxiou_per_gt,extra_anchors]
			maxiou_per_gt = np.concatenate(indexes,axis=0)

			labels = np.concatenate([labels,extra_labels],axis=0)

	# mask generation
	gt_mask = np.zeros((iou.shape[0],4))
	# only indexes maxiou_per_gt are valid bounding boxes 
	'''
	????????????????????? 1.0 ??????????????????????????????????????? 
	????????????ground truth bounding boxes???anchor boxes???
	?????????positive anchor boxes???extra positive anchor boxes???
	???????????????ground truth mask????????????
	'''
	gt_mask[maxiou_per_gt] = 1.0 

	'''??????????????????????????????????????????????????????'''
	# classes generation
	gt_class = np.zeros((iou.shape[0],n_classes))
	# by default all are background (index 0)
	gt_class [:,0] =1 

	'''??????positive????????????????????????????????????????????????'''
	
	# but those that belong to maxiou_per_gt are not 
	gt_class[maxiou_per_gt,0] = 0 
	
	# we have to find those column indexes(classes)
	maxiou_col = np.reshape(maxiou_per_gt,(maxiou_per_gt.shape[0],1))
	label_col = np.reshape(labels[:4],(labels.shape[0],1)).astype(int)

	row_col = np.append(maxiou_col,label_col,axis=1)
	# the label of object in maxiou_per_gt 
	gt_class[row_col[:,0],row_col[:,1]] = 1.0 

	# offsets generation
	gt_offset = np.zeros((iou.shape[0],4))

	# (cx,cy,w,h) format 
	if normalize:
		anchors = minmax2centroid(anchors)
		labels = minmax2centroid(labels)

		# bbox = bouding boxes
		# ((bbox xcenter - anchor box x center)/anchor box width)/.1
		# ((bbox ycenter - anchor box y center)/anchor box height)/.1
		""" Equation 11.4.8 chapter11"""
		offset2 = np.log(labels[:,2:4]/anchors[maxiou_per_gt,2:4])
		offset2 /= 0.2
		
		offsets = np.concatenate([offset1,offset2],axis=-1)

	# (xmin,xmax,ymin,ymax) format 
	else:
		offsets = labels[:,0:4] - anchors[maxiou_per_gt]

	gt_offset[maxiou_per_gt] = offsets
	return gt_class,gt_offset,gt_mask




def minmax2centroid(boxes):
	'''
	MinMax to centroid format
	(xmin,xmax,ymin,ymax) to (cx,cy,w,h)
	Arguments:
		boxes(tensor):	   Batch of boxes in minmax format
	Returns:
		centroid(tensor):  Batch of boxes in centroid format
	'''
	centroid = np.copy(boxes).astype(np.float)
	centroid[...,0] = 0.5 *(boxes[...,1] - boxes[...,0])
	centroid[...,0] += boxes[...,0]

	centroid[...,1] = 0.5*(boxes[...,3] - boxes[...,2])
	centroid[...,1] += boxes[...,2]

	centroid[...,2] = boxes[...,1] - boxes[...,0]
	centroid[...,3] = boxes[...,3] - boxes[...,2]

	return centroid





def intersection(boxes1,boxes2):
	'''Compute intersection of batch of boxes1 and boxes2
	Arguments:
		boxes1(tensor):Boxes coordinates in pixels
		boxes2(tensor):Boxes coordinate in pixels
	Returns:
		intersection_areas (tensor): intersection of areas of boxes1 and boxes2 
	'''
	m = boxes1.shape[0]
	n = boxes2.shape[0]

	xmin = 0 
	xmax = 1
	ymin = 2
	ymax = 3

	boxes1_min = np.expand_dims(boxes1[:,[xmin,ymin]],axis=1)
	boxes1_min = np.tile(boxes1_min,reps=(1,n,1))
	boxes2_min = np.expand_dims(boxes2[:,[xmin,ymin]],axis=0)
	boxes2_min = np.tile(boxes2_min,reps=(m,1,1))
	min_xy = np.maximum(boxes1_min,boxes2_min)

	boxes1_max = np.expand_dims(boxes1[:,[xmax,ymax]],axis=1)
	boxes1_max = np.tile(boxes1_max,reps=(1,n,1))
	boxes2_max = np.expand_dims(boxes2[:,[xmax,ymax]],axis=0)
	boxes2_max = np.tile(boxes2_max,reps=(m,1,1))
	max_xy = np.minimum(boxes1_max,boxes2_max)

	side_lengths = np.maximum(0,max_xy - min_xy)
	intersection_areas = side_lengths[:,:,0] *side_lengths[:,:,1]
	return intersection_areas


def union(boxes1,boxes2,intersection_areas):
	'''Compute union of batch of boxes1 and boxes 2
	Arguments:
		boxes1(tensor): Boxes coordinates in pixels
		boxes2(tensor): Boxes coordinates in pixels

	
	Returns:
		union_areas(tensor): union of areas of boxes1 and boxes2 
	'''
	m = boxes1.shape[0]
	n = boxes2.shape[0]

	xmin = 0
	xmax = 1 
	ymin = 2 
	ymax = 3

	width = (boxes1[:,xmax] - boxes1[:,xmin])
	height = (boxes1[:,ymax] - boxes1[:,ymin])
	areas = width*height
	
	boxes1_areas = np.tile(np.expand_dims(areas,axis=1),reps=(1,n))
	width = (boxes2[:,xmax]- boxes2)
	height = (boxes1[:,ymax] - boxes1[:,ymin])
	areas = width *height
	boxes1_area = np.tile(np.expand_dims(areas,axis=1),reps=(1,n))
	width = (boxes2[:,xmax] - boxes2[:,xmin])
	height = (boxes2[:,ymax] - boxes2[:,ymin])
	areas = width * height
	boxes2_areas = np.tile(np.expand_dims(areas,axis=0),reps=(m,1))

	union_areas = boxes1_area + boxes2_areas - intersection_areas
	return union_areas

def iou(boxes1,boxes2):
	'''Compute IoU of batch boxes1 and boxes2
	Arguments:
		boxes1(tensor): Boxes coordinates in pixels
		boxes2(tensor): Boxes coordinates in pixels
	Returns:
		iou(tensor): intersection of union of areas of boxes1 and boxes2
	'''
	intersection_areas =intersection(boxes1,boxes2)
	union_areas = union(boxes1,boxes2,intersection_areas)
	return intersection_areas/union_areas

