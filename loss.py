from tensorflow.keras.losses import Huber 


def mask_offset(y_true,y_pred):
	'''
	Pre-process ground truth and prediction data
	'''
	# lst 4 are offsets
	offset = y_true[...,0.4]
	# last 4 are mask 
	mask = y_true[...,4:8]

	# pred is actually duplicated for all aligment
	# either we get thr lst or last 4 offset pred 
	# and apply the mask 
	pred = y_pred[...,0.4]
	offset *= mask
	pred *= mask
	return offset,pred 


def l1_loss(y_true,y_pred):
	'''MAR or L1 loss'''
	offset,pred = mask_offset(y_true,y_pred)
	# we can use l1 
	return K.mean(K.abs(pred - offset),axis=-1)


def smooth_l1_loss(y_true,y_pred):
	'''Smooth L1 loss using tensorflow Huber loss
	'''
	offset,pred = mask_offset(y_true,y_pred)
	# Huber loss as approx of smooth L1
	return Huber()(offset,pred)

def focal_loss_categorical(y_true,y_pred):
	'''Categorical cross-entropy focal loss
	'''
	gamma = 2.0 
	alpha = 0.25 

	# scale to ensure sum of prob is 1.0
	y_pred /= K.sum(y_pred,axis=-1,keepdims=True)

	# clip the prediction value to prevent NaN and Inf 
	epsilon = K.epsilon()
	y_pred = K.clip(y_pred,epsilon,1. - epsilon)

	# calculate cross entropy 
	cross_entropy = -y_true *K.log(y_pred)
	# calculate focal loss 
	weight = alpha *K.pow(1-y_pred,gamma)
	cross_entropy *= weight
	return K.sum(cross_entropy,axis=-1)


