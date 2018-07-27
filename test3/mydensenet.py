from keras.models import Model
from keras.layers import Input, add, LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.convolutional import Conv2D
import keras.backend as K
import tensorflow as tf

def DenseNet(nb_dense_block=20, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weights_path=None):
    '''Instantiate the DenseNet 121 architecture,
        # Arguments
            nb_dense_block: number of dense blocks to add to end
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate            
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''
    
    # compute compression factor
    compression = 1.0 - reduction

    
    img_input = Input(shape=(None, None, 4), name='data')
    

    # From architecture for ImageNet (Table 1 in the paper)
    #nb_filter = 64
    nb_layers = 3 # For DenseNet-121

    # Initial convolution

    x = Conv2D(12, (1, 1), padding='same', kernel_initializer='glorot_normal', name='conv0')(img_input)
    x = LeakyReLU(alpha=0.1)(x)
    GLR = x
    x = Conv2D(nb_filter, (3, 3), padding='same', kernel_initializer='glorot_normal', name='conv1')(x)
    x = LeakyReLU(alpha=0.1)(x)


    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        LR = x
        x = dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)
        
        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate)
        x = add([LR, x])
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x = dense_block(x, final_stage, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate)
    x = Conv2D(12, (1, 1), padding='same', kernel_initializer='glorot_normal')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = add([x, GLR])
    x = SubpixelConv2D(input_shape = x.shape, scale = 2)(x)

    model = Model(img_input, x, name='densenet')

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate            
    '''
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = Conv2D(inter_channel, (1, 1), padding='same', kernel_initializer='glorot_normal', name=conv_name_base+'_x1')(x)
    x = LeakyReLU(alpha=0.1)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = Conv2D(nb_filter, (3, 3), padding='same', kernel_initializer='glorot_normal', name=conv_name_base+'_x2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate            
    '''

    
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    
    x = Conv2D(int(nb_filter * compression), (1, 1), padding='same', kernel_initializer='glorot_normal', name=conv_name_base)(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
  
    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate            
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
        concat_feat = concatenate([concat_feat, x], name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat

def SubpixelConv2D(input_shape, scale=2):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1],
                input_shape[2],
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)


    return Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')

