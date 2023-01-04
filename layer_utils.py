
import numpy as np

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
	maxiou_per_gt = np.argmax(iou,axis=0)

	# get extra anchor boxes based on IoU
	if threshold<1.0:
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
	gt_mask[maxiou_per_gt] = 1.0 

	# classes generation
	gt_class = np.zeros((iou.shape[0],n_classes))
	# by default all are background (index 0)
	gt_class [:,0] =1 
	# but those that belong to maxiou_per_gt are not 
	gt_class[maxiou_per_gt,0] = 0 
	

	