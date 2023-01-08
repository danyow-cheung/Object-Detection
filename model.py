
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import ELU, MaxPooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np 


def build_ssd(input_shape,backone,n_layers = 4,n_classes = 4,aspect_ratios=(1,2,0.5)):
	'''Build SSD model given a backone
	Arguments:
		input_shape(list):  input_image shape 
		backone(model):     Keras backone model
		n_layers(int):		Number of layers of ssd head 
		n_classes(int):		Number of obj classes 
		aspect_ratios(list):annchor box aspect ratios
	Returns:
		n_anchors(int):		   Number of anchor boxes per feature pt 
		feature_shape(tensor): SSD head feature maps 
		model(Keras model)   : SSD model
		'''
	# number of anchor boxes per feature map pt 
	n_anchors = len(aspect_ratios)+1 

	inputs = Input(shape=input_shape)
	# number of base_outputs depends on n_layers 
	base_outputs = backone(inputs)

	outputs = []

	feature_shapes = []
	out_cls = []
	out_off = []

	for i in range(n_layers):
		# each conv layer from backone is used 
		# as feature maps for class and offset predictions
		# also known as multi-scale predictions
		conv= base_outputs if n_layers==1 else base_outputs[i]

		name ="cls"+str(i+1)
		classes = conv2d(conv,n_anchors*n_classes,kerenl_size =3,name=name)

		# offsets :(batch,height,width,n_anchors*4)
		name = "off"+str(i+1)
		offsets = conv2d(conv,n_anchors*4,kernel_size=3,name=name)

		shape = np.array(K.int_shape(offsets))[1:]
		feature_shapes.append(shape)

		# reshape the class predictions ,yielding 3D tensors of 
		# shape (batch, height*width*n_anchors, n_classes)
		# last axis to perform softmax on them
		name = "cls_res" +str(i+1)

		classes = Reshape((-1,n_classes),name=name)(classes)

		# reshape the offset predictions ,yielding 3D tensor of 
		# shape  (batch, height*width*n_anchors, 4)
		# last axis to compute the(smooth) l1 or l2 loss 
		name = "off_res" +str(i+1)
		offsets = Reshape((-1,4),name=name)(offsets)

		# concat for aligment with ground truth size 
		# made of ground truth offsets and mask of same dim
		# needed during loss computation
		offsets = [offsets,offsets]
		name = "off_cat"+str(i+1)
		offsets = Concatenate(axis=-1,name=name)(offsets)

		# collect offset prediction per scale 
		out_off.append(offsets)

		name = "cls_out" + str(i+1)

		# activation = 'sigmoid' if n_classess ==1 else "softmax"
		classes = Activation('softmax',name=name)(classes)

		# collect class prediction per scale 
		out_cls.append(classes)
	
	if n_layers>1:
		# concat all class and offset from each scale 
		name = 'offsets'
		offsets = Concatenate(axis=1,name=name)(out_off)

		name = 'classes'
		classes = Concatenate(axis=1,name=name)(out_cls)

	else:
		offsets = out_off[0]
		classes = out_cls[0]
	
	outputs = [classes,offsets]
	model = Model(inputs= inputs,outputs=outputs,name='ssd-head')

	return n_anchors,feature_shapes,model
	