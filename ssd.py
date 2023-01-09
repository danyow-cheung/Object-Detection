import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import Huber

import layer_utils
import label_utils
import config

import os
import skimage
import numpy as np
import argparse

from skimage.io import imread
from data_generator import DataGenerator
from label_utils import build_label_dictionary
from boxes import show_boxes
from model import build_ssd
from loss import focal_loss_categorical, smooth_l1_loss, l1_loss
from model_utils import lr_scheduler, ssd_parser
from common_utils import print_log


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


	def build_generator(self):
		'''Build a multi-thread train data generator'''
		self.train_generator = DataGenerator(
			args=self.args,
			dictionary=self.dictionary
			n_classes = self.n_classes,
			features_shapes = self.feature_shapes,
			n_anchors = self.n_anchors,
			shuffle=True
		)

	def train(self):
		'''train an ssd network'''
		# build the train data generator
		if self.train_generator is None:
			self.build_generator()

		optimzer = Adam(lr=1e-3)
		# choice of loss function via args
		if self.args.improved_loss:
			print_log("Focal loss and smooth L1",self.args.verbose)
			loss = [focal_loss_categorical,smooth_l1_loss]
		elif self.args.smooth_l1:
			print_log("Smooth L1 ",self.args.verbose)
			loss = ['categorical_crossentropy',smooth_l1_loss]
		else:
			print_log("Cross-entropy and L1",self.args.verbose)
			loss = ['categorical_crossentropy',l1_loss]
		
		self.ssd.compile(optimzer=optimzer,loss=loss)

		# prepare callbacks for saving model weights
		# and learning rate scheduler
		# learning rate decreases by 50% every 20 epochs
		# after 60th epoch 
		checkpoint = ModelCheckpoint(filepath=filepath,verbose=1,save_weights_only=True)

		scheduler = LearningRateScheduler(lr_scheduler)

		callbacks = [checkpoint,scheduler]

		# train the ssd network
		self.ssd.fit_generator(generator=self.train_generator,
								use_multiprocessing =True,
								callbacks=callbacks,
								epochs = self.args.epochs, 
								workers = self.args.workers
								)

	def evaluate_test(self):
		# test labels csv path
		path = os.path.join(self.args.data_path,self.args.test_labels)
		# test dictionary
		dictionary ,_ =  build_label_dictionary(path)
		keys = np.array(list(dictionary.keys()))
		# sum of precision
		s_precision = 0 
		# sum of recall 
		s_recall = 0 
		# sum of IoUs 
		s_iou = 0 
		# evaluate per image 
		for key in keys:
			# ground truth labels
			labels = np.array(dictionary[key])
			# 4 boxes coords are 1st four items of labels
			gt_boxes = labels[:,0:-1]
			# last one is class 
			gt_class_ids = labels[:,-1]
			# load image id by key 
			image_file = os.path.join(self.args.data_path,key)
			image = skimage.img_as_float(imread(image_file))
			image ,classes ,offsets = self.detect_objects(image)
			# perform nms
			_,_,class_ids ,boxes = show_boxes(
				self.args,
				image,
				classes,
				offsets,
				self.feature_shapes,
				show = False 
			)
			boxes = np.reshape(np.array(boxes),(-1,4))
			# compute IoUs 
			iou = label_utils.iou(gt_boxes,boxes)
			# skip empty IoUs
			if iou.size ==0:
				continue
			# the class of predicted box w/max iou 
			maxiou_class = np.argmax(iou,axis=1)

			# true positive
			tp = 0 
			# false postive 
			fp = 0 
			# sum of objects iou per image 
			s_image_iou = []
			for n in range(iou.shape[0]):
				# ground truth bbox has a label
				if iou[n,maxiou_class[n]]>0:
					s_image_iou.append(iou[n,maxiou_class[n]])
					# true postivie has the same class and gt 
					if gt_class_ids[n] == class_ids[maxiou_class[n]]:
						tp+=1
					else:
						fP+=1
			# objects that we missed(false negative)
			fn = abs(len(gt_class_ids) -tp)
			s_iou += (np.sum(s_image_iou) / iou.shape[0])
			s_precision += (tp/(tp+fp))
			s_recall += (tp/(tp+fn))

		n_test = len(keys)
		print_log("mIoU: %f" % (s_iou/n_test),self.args.verbose)
		print_log("Precision: %f" % (s_precision/n_test),self.args.verbose)
		print_log("Recall: %f" % (s_recall/n_test),self.args.verbose)

