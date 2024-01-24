import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
import numpy as np

class GELU(layers.Layer):
    def __init__(self):
        super(GELU, self).__init__()

    def call(self, x):
        cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf
 
def pre_Conv(inputs, out_channel):
 
    x = layers.Conv2D(filters=out_channel,
                      kernel_size=(2,2),
                      strides=2, 
                      padding='same')(inputs)
 
    x = layers.LayerNormalization()(x)
 
    return x
 
 
# ConvNeXt Block
def block(inputs, dropout_rate=0.2, layer_scale_init_value=1e-6):

    dim = inputs.shape[-1]

    residual = inputs

    x = layers.DepthwiseConv2D(kernel_size=(7,7), strides=1, padding='same')(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Conv2D(filters=dim*4, kernel_size=(1,1), strides=1, padding='same')(x)
    x = GELU().call(x)
    x = layers.Conv2D(filters=dim, kernel_size=(1,1), strides=1, padding='same')(x)
    
    gama = layers.Layer().add_weight(shape=[dim], 
                                   initializer=tf.initializers.Constant(layer_scale_init_value), 
                                   dtype=tf.float32, 
                                   trainable=True) 
 
    x = x * gama  # [56,56,96]*[96]==>[56,56,96]
 
    x = layers.Dropout(rate=dropout_rate)(x)
 
    x = layers.add([x, residual])
    
    return x
    

def downsampling(inputs, out_channel):
 
    x = layers.LayerNormalization()(inputs)
    
    x = layers.Conv2D(filters=out_channel, 
                      kernel_size=(2,2),
                      strides=2, 
                      padding='same')(x)
    
    return x
 
 
def stage(x, num, out_channel, downsampe=True):
    if downsampe is True:
        x = downsampling(x, out_channel)
 
    for _ in range(num):
        x = block(x)
 
    return x
 

def Convnext(input_shape, classes): 
 
    # [512,512,3]==>[256,256,64]
    x = pre_Conv(input_shape, out_channel=64)
    feat1 = x
    # [256,256,64]==>[128,128,256]
    x = stage(x, num=3, out_channel=256, downsampe=True)
    feat2 = x
    # [128,128,256]==>[64,64,512]
    x = stage(x, num=3, out_channel=512, downsampe=True)
    feat3 = x
    # [64,64,512]==>[32,32,1024]
    x = stage(x, num=9, out_channel=1024, downsampe=True)
    feat4 = x
    # [32,32,1024]==>[16,16,2048]
    x = stage(x, num=3, out_channel=2048, downsampe=True)
    feat5 = x
 
    # [7,7,768]==>[None,768]
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.LayerNormalization()(x)
 
    # [None,768]==>[None,classes]
    # logits = layers.Dense(classes)(x)  

    # model = Model(inputs, logits)
 
    # return model
    return feat1, feat2, feat3, feat4, feat5
