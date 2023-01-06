import os
import numpy as np

class SSD(object):
	"""
	Made of an ssd network model and a dataset generator
	SSD defines functions to train and validate
	and ssd network model
	
	Arguments:
		args:User-defined config
	Attributes:
		ssd(model) : SSD network model
		train_generator: Multi-threaded threaded data generator for training
	"""

	def __init__(self, arg):

		super(SSD, self).__init__()
		self.args = args
		self.ssd = None 
		self.train_generator = None
		self.build_model()

	def build_model(self):
		'''Build backone and SSD models'''
		# store in a dictionary the list of image files and labels
		self.build_dictionary()

		# input shape is (480,640,3) by default
		self.input_shape = (self.args.height,self.args.width,self.args.channels)

		# build the backone network (eg ResNet50)
		# the number of feature layers is equal to n_layers
		# feature layers are inputs to SSD network heads
		# for class and offsets predictions
		self.backone = self.args.backone(self.input_shape,n_layers= self.args.layers)

		# using the backone ,build ssd network
		# outputs of ssd are class and offset predictions
		anchors,features,ssd = build_ssd(self.input_shape,self.backone,n_layers=self.args.layers,n_classes = self.n_classes)

		# n_anchors = num of anchors per feature point(eg 4)
		self.n_anchors = anchors

		# feature_shapes is a list of feature map shape
		# per output layer -- used for computing anchor boxes sizes
		self.feature_shapes = features
		# ssd network model
		self.ssd = ssd 


	def build_dictionary(self):
		'''Read input image filename and obj detection labels
		from a csv file and store in a dictionary
		'''
		# train dataset path
		path = os.path.join(self.args.data_path,self.args.train_labels)

		# build dict
		# key = image filename ,value = box coords + class label
		# self.classes is a list of class labels
		self.dictionary ,self.classes = build_label_dictionary(path)

		self.n_classes = len(self.classes)

		self.keys = np.array(list(self.dictionary.keys()))