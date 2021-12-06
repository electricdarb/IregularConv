import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import models

Conv2D_ = Conv2D
    
class IdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, stage, block, Conv2D = Conv2D,
     reg = tf.keras.regularizers.L2(0.0001),
     wpk = 4):
        """The identity block is the block that has no conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        # Returns
            Output tensor for the block.
        """
        super(IdentityBlock, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stage = stage
        self.block = block 
        self.Conv2D = Conv2D
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv1 = Conv2D_(filters1, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a', kernel_regularizer=reg)
        self.bn1 = BatchNormalization(name=bn_name_base + '2a')

        self.conv2 = Conv2D(filters2, kernel_size,
                        padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b', kernel_regularizer=reg) 
        self.conv2.wpk = wpk
        self.bn2 = BatchNormalization(name=bn_name_base + '2b')

        self.conv3 = Conv2D_(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c', kernel_regularizer=reg)
        self.bn3 = BatchNormalization(name=bn_name_base + '2c')
    
    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = Activation('relu')(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = Activation('relu')(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x

class ConvBlock(tf.keras.Model):
    def __init__(self, 
                kernel_size,
                filters,
                stage,
                block,
                strides=(2, 2), 
                Conv2D = Conv2D,
                reg = tf.keras.regularizers.L2(0.0001), 
                wpk = 4):
        """A block that has a conv layer at shortcut.
        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.
        # Returns
            Output tensor for the block.
        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        super(ConvBlock, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stage = stage
        self.block = block 
        self.strides = strides
        self.Conv2D = Conv2D

        filters1, filters2, filters3 = filters
        
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv1 = Conv2D_(filters1, (1, 1), strides=strides,
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a',  kernel_regularizer=reg)
        self.bn1 = BatchNormalization(name=bn_name_base + '2a')

        self.conv2 = Conv2D(filters2, kernel_size, padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b',  kernel_regularizer=reg) 
        self.conv2.wpk = wpk
        self.bn2 = BatchNormalization(name=bn_name_base + '2b')

        self.conv3 = Conv2D_(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c', kernel_regularizer=reg)
        self.bn3 = BatchNormalization(name=bn_name_base + '2c')

        self.conv4 =  Conv2D_(filters3, (1, 1), strides=strides,
                                kernel_initializer='he_normal',
                                name=conv_name_base + '1', kernel_regularizer=reg)
        self.bn4 = BatchNormalization(name=bn_name_base + '1')

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = Activation('relu')(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = Activation('relu')(x)

        x = self.conv3(x)
        x = self.bn3(x)

        shortcut = self.conv4(input_tensor)
        shortcut = self.bn4(shortcut)
        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x

def ResNet50(input_shape,
 Conv2D = Conv2D_, 
 classes = 10, 
 reg = tf.keras.regularizers.L2(0.0001),
 weights_per_kernel = 4):
    # Determine proper input shape

    input_tensor = Input(shape = input_shape)

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_tensor)
    x = Conv2D_(64, (7, 7),
                    strides=(2, 2),
                    padding='valid',
                    kernel_initializer='he_normal',
                    name='conv1',  kernel_regularizer=reg)(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = ConvBlock(3, [64, 64, 256], stage=2, block='a', strides=(1, 1), Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [64, 64, 256], stage=2, block='b', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [64, 64, 256], stage=2, block='c', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)

    x = ConvBlock(3, [128, 128, 512], stage=3, block='a', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [128, 128, 512], stage=3, block='b', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [128, 128, 512], stage=3, block='c', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [128, 128, 512], stage=3, block='d', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)

    x = ConvBlock(3, [256, 256, 1024], stage=4, block='a', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [256, 256, 1024], stage=4, block='b', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [256, 256, 1024], stage=4, block='c', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [256, 256, 1024], stage=4, block='d', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [256, 256, 1024], stage=4, block='e', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [256, 256, 1024], stage=4, block='f', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)

    x = ConvBlock(3, [512, 512, 2048], stage=5, block='a', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [512, 512, 2048], stage=5, block='b', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)
    x = IdentityBlock(3, [512, 512, 2048], stage=5, block='c', Conv2D = Conv2D, reg = reg, wpk = weights_per_kernel)(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    y = Dense(classes, activation='softmax', name='fc1000', kernel_regularizer=reg)(x)

    model = models.Model(inputs=input_tensor, outputs=y, name="ResNet50")
    return model

