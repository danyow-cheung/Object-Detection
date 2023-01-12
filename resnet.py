from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.layers import Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np

from model import conv_layer


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    '''2d convolution-batch Normalization-Activation stack builder
    Arguments:
        inputs(tensor):             Input tensor from input image or previous layer
        num_filets(int):            Conv2D number of filters

        kernel_size(int):           Conv2D square kernel dimensions
        strides(int):               Conv2D square stride dimensions
        activation(string):         Activation name
        batch_normalization(bool):  Whether to include batch normalization
        conv_first(bool):           Conv-bn-activation(True) or bn-activation-conv(False)
    Returns:
        x(tensor):                  Tensor as input to the next layer
    '''
    conv = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))

    x = inputs 
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x 

def resnet_v1(input_shape,depth,num_classes=10):
    '''ResNet v1 model builder
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    
    Arguments:
        input_shape(tensor):    Shape of input image tensor
        depth(int):             Number of core convolutional layers
        num_classes(int):       Number of classess
    Returns:
        model(Model):           Keras model instance
    '''
    if (depth-2)%6!=0:
        raise ValueError("depth shoule be 6n+2(eg 20,32,44 in [a])")
    # start model deinition 
    num_filters = 16 
    num_res_blocks = int((depth-2)/6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # instantiate the stack of residual units 
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1 
            # first layer but not first stack
            if stack>0 and res_block==0:
                strides =2 
            y = resnet_layer(inputs=x,num_filters=num_filters,strides=strides)

            y = resnet_layer(inputs = y, num_filters=num_filters,activation=None)
            # first layer but not first stack 
            if stack>0 and res_block==0:
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x,num_filters=num_filters,kernel_size=1,strides=strides,activation=None,batch_normalization=False)
            
            x = Add()([x,y])
            x = Activation('relu')(x)
        num_filters*=2 
        
    # 1st feature map layer 
    conv = AveragePooling2D(pool_size=4,name='pool1')(x)

    outputs = [conv]
    prev_conv = conv 
    n_filters  = 64

    # additional feature map layers
    for i in range(n_layers -1):
        postfix = "_layer"+str(i+2)
        conv = conv_layer(prev_conv,n_filters,kernel_size=3,strides=2,use_maxpool=False,postfix=postfix)

        outputs.append(conv)
        prev_conv = conv 
        n_filters *= 2
    
    # instantiate model

    name = 'ResNet%dv1' % (depth)
    model = Model(inputs=inputs,
                  outputs=outputs,
                  name=name)
    return model



def resnet_v2(input_shape,depth,n_layers=4):
    '''ResNet Versioin 2 Model builder
    
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    
    Arguments
        input_shape (tensor):       Shape of input image tensor
        depth (int):                Number of core convolutional layers
        num_classes (int):          Number of classes (CIFAR10 has 10)
    Returns
        model (Model):              Keras model instance
    '''
    if (depth-2)%9!=0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16 
    num_res_blocks = int((depth-2)/9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-RELU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,num_filters=num_filters_in,conv_first=True)
    # Instantiate the stack of residual units 
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides =1 
            if stage==0:
                num_filters_out = num_filters_in*4 
                # first layer and first stage
                if res_block==0:
                    activation=None
                    batch_normalization = False 
            else:
                num_filters_out = num_filters_in*2 
                # first layer but not first stage
                if res_block==0 :
                    strides=2# downsample 
            
            # bottleneck residual unit 
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False)
            
            y = resnet_layer(inputs=y,
                            num_filters=num_filters_in,
                            conv_first=False)
            y = resnet_layer(inputs=y,
                            num_filters=num_filters_out,
                            kernel_size=1,
                            conv_first=False)
            if res_block==0:
                #linear project residual shorcut connection to match changed dims 
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = Add()([x,y])
        num_filters_in = num_filters_out
    # v2 have BN-RELU before pooling 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # 1st feature map layer
    conv = AveragePooling2D(pool_size=4,name='pool1')(x)

    outputs = [conv]

    prev_conv = conv 

    n_filters = 64 

    # additional feature map layers
    for i in range(n_layers - 1):
        postfix = "_layer" + str(i+2)
        conv = conv_layer(prev_conv,
                          n_filters,
                          kernel_size=3,
                          strides=2,
                          use_maxpool=False,
                          postfix=postfix)
        outputs.append(conv)
        prev_conv = conv
        n_filters *= 2
    

    # instantiate model.
    name = 'ResNet%dv2' % (depth)
    model = Model(inputs=inputs,
                  outputs=outputs,
                  name=name)
    return model

def build_resnet(input_shape,n_layers,version=2,n=6):
    '''Build a resnet as backone of SSD
    Arguments:
        input_shape(list):          Input image size and channels
        n_layers(int):              Number of feature layers for SSD 
        version(int):               Supports ResNetv1 and v2 but v2 by default
        n(int):                     Determines number of ResNet layers (Default is ResNet50)
    Returns:
        model(keras model)
    '''
    # computed depth from supplied model parameter n
    if version ==1:
        depth = n*6 + 2
    elif version==2:
        depth = n*9 +2 
    
    # model name,depth and version
    # input_shape(h,w,3)
    if version==1:
        model = resnet_v1(input_shape=input_shape,depth=depth,n_layers=n_layers)
    else:
        model = resnet_v2(input_shape=input_shape,depth=depth,n_layers=n_layers)
    return model 
    


