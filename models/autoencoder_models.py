import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Flatten,\
                                    Conv2DTranspose, UpSampling2D, Reshape, LayerNormalization,\
                                    BatchNormalization
from tensorflow.keras.backend import l2_normalize
import pdb

class conv4_fc1:
    def __init__(self, flags, input_shape):
        self.flags= flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.build_model()

    def build_model(self):
        flags= self.flags
        self.l2_regul= tf.keras.regularizers.l2(l= flags.l2_regul)

        input_tensor= tf.keras.Input(shape= self.input_shape)
        x= Conv2D(filters= flags.filters1,   kernel_size= flags.first_kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv1')(input_tensor)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters2,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= flags.filters3,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv3')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters4,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        e_pool2= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        e_flatten= Flatten()(e_pool2)

        embedding= Dense(units= flags.embedding_size,  activation= None, name= 'e_fc')(e_flatten)
        embedding= l2_normalize(embedding)
        self.embedding= embedding
        
        d_flatten= Dense(units= e_flatten.shape[1],    activation= None, name= 'd_fc')(embedding)
        reshaped= Reshape(target_shape= e_pool2.shape[1:])(d_flatten)
        x= UpSampling2D(size= flags.pool_size)(reshaped)
        x= Conv2DTranspose(filters= flags.filters3,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 strides= flags.conv_strides,
                                 padding= flags.conv_padding, name= 'd_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 strides= flags.conv_strides,
                                 padding= flags.conv_padding, name= 'd_conv3')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 strides= flags.conv_strides,
                                 padding= flags.conv_padding, name= 'd_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        output_tensor= Conv2DTranspose(filters= 1,  kernel_size= flags.first_kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 strides= flags.conv_strides,
                                 padding= flags.conv_padding, name= 'output_tensor')(x)

        self.model= tf.keras.Model(inputs= input_tensor, outputs= output_tensor)
        self.optimizer= tf.keras.optimizers.Adam(flags.learning_rate)
        self.loss_object= tf.keras.losses.MeanSquaredError()


class conv4_fc2:
    def __init__(self, flags, input_shape):
        self.flags= flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.build_model()

    def build_model(self):
        flags= self.flags
        self.l2_regul= tf.keras.regularizers.l2(l= flags.l2_regul)

        input_tensor= tf.keras.Input(shape= self.input_shape)
        x= Conv2D(filters= flags.filters1,   kernel_size= flags.first_kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv1')(input_tensor)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters2,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= 'same')(x)
        x= Conv2D(filters= flags.filters3,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv3')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters4,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        e_pool2= MaxPool2D(pool_size= flags.pool_size, padding= 'same')(x)
        e_flatten= Flatten()(e_pool2)
        e_fc1= Dense(units= flags.units1,  activation= flags.act_fn1, name= 'e_fc1')(e_flatten)

        embedding= Dense(units= flags.embedding_size,  activation= None, name= 'e_fc2')(e_fc1)
        embedding= l2_normalize(embedding)
        self.embedding= embedding
        
        d_fc2= Dense(units= flags.units1, activation= flags.act_fn1, name= 'd_fc2')(embedding)
        d_fc1= Dense(units= e_flatten.shape[1],    activation= None, name= 'd_fc1')(d_fc2)
        reshaped= Reshape(target_shape= e_pool2.shape[1:])(d_fc1)
        d_uppool2= UpSampling2D(size= flags.pool_size)(reshaped)
        x= Conv2DTranspose(filters= flags.filters3,  kernel_size= flags.kernel_size,
                           activation= flags.act_fn1, dtype= tf.float32,
                           kernel_regularizer= self.l2_regul, 
                           strides= flags.conv_strides,
                           padding= flags.conv_padding, name= 'd_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                           activation= flags.act_fn1, dtype= tf.float32,
                           kernel_regularizer= self.l2_regul, 
                           strides= flags.conv_strides,
                           padding= flags.conv_padding, name= 'd_conv3')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                           activation= flags.act_fn1, dtype= tf.float32,
                           kernel_regularizer= self.l2_regul, 
                           strides= flags.conv_strides,
                           padding= flags.conv_padding, name= 'd_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        output_tensor= Conv2DTranspose(filters= 1,  kernel_size= flags.first_kernel_size,
                                       activation= flags.act_fn1, dtype= tf.float32,
                                       kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                                       padding= flags.conv_padding, name= 'output_tensor')(x)

        self.model= tf.keras.Model(inputs= input_tensor, outputs= output_tensor)
        self.optimizer= tf.keras.optimizers.Adam(flags.learning_rate1)
        self.loss_object= tf.keras.losses.MeanSquaredError()

class conv6_fc1:
    def __init__(self, flags, input_shape):
        self.flags= flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.build_model()

    def build_model(self):
        flags= self.flags
        self.l2_regul= tf.keras.regularizers.l2(l= flags.l2_regul)

        input_tensor= tf.keras.Input(shape= self.input_shape)
        x= Conv2D(filters= flags.filters1,   kernel_size= flags.first_kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv1')(input_tensor)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters2,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= flags.filters3,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv3')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters4,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= flags.filters5,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv5')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters6,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                        padding= flags.conv_padding, name= 'e_conv6')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        e_pool3= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        e_flatten= Flatten()(e_pool3)

        embedding= Dense(units= flags.embedding_size,  activation= None, name= 'e_fc')(e_flatten)
        embedding= l2_normalize(embedding)
        self.embedding= embedding
        
        d_flatten= Dense(units= e_flatten.shape[1],    activation= None, name= 'd_fc')(embedding)
        reshaped= Reshape(target_shape= e_pool3.shape[1:])(d_flatten)
        d_uppool3= UpSampling2D(size= flags.pool_size)(reshaped)
        x= Conv2DTranspose(filters= flags.filters5,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                                 padding= flags.conv_padding, name= 'd_conv6')(d_uppool3)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= flags.filters4, kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                                 padding= flags.conv_padding, name= 'd_conv5')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= flags.filters3,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                                 padding= flags.conv_padding, name= 'd_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                                 padding= flags.conv_padding, name= 'd_conv3')(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                       strides= flags.conv_strides,
                                 padding= flags.conv_padding, name= 'd_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        output_tensor= Conv2DTranspose(filters= 1,  kernel_size= flags.first_kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'output_tensor')(x)

        self.model= tf.keras.Model(inputs= input_tensor, outputs= output_tensor)
        self.optimizer= tf.keras.optimizers.Adam(flags.learning_rate1)
        self.loss_object= tf.keras.losses.MeanSquaredError()

class conv6_fc2:
    def __init__(self, flags, input_shape):
        self.flags= flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.build_model()

    def build_model(self):
        flags= self.flags
        self.l2_regul= tf.keras.regularizers.l2(l= flags.l2_regul)

        input_tensor= tf.keras.Input(shape= self.input_shape)
        e_conv1= Conv2D(filters= flags.filters1,   kernel_size= (self.num_rows, 3),
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv1')(input_tensor)
        e_conv2= Conv2D(filters= flags.filters2,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv2')(e_conv1)
        e_pool1= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(e_conv2)
        e_conv3= Conv2D(filters= flags.filters3,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv3')(e_pool1)
        e_conv4= Conv2D(filters= flags.filters4,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv4')(e_conv3)
        e_pool2= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(e_conv4)
        e_conv5= Conv2D(filters= flags.filters5,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv5')(e_pool2)
        e_conv6= Conv2D(filters= flags.filters6,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv6')(e_conv5)
        e_pool3= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(e_conv6)
        e_flatten= Flatten()(e_pool3)
        e_fc1= Dense(units= flags.units1,  activation= flags.act_fn1, name= 'e_fc1')(e_flatten)

        embedding= Dense(units= flags.embedding_size,  activation= None, name= 'e_fc2')(e_fc1)
        embedding= l2_normalize(embedding)
        self.embedding= embedding
        
        d_fc2= Dense(units= flags.units1,  activation= flags.act_fn1, name= 'd_fc2')(embedding)
        d_flatten= Dense(units= e_flatten.shape[1],    activation= None, name= 'd_fc1')(d_fc2)
        reshaped= Reshape(target_shape= e_pool3.shape[1:])(d_flatten)
        d_uppool3= UpSampling2D(size= flags.pool_size)(reshaped)
        d_conv6= Conv2DTranspose(filters= flags.filters5,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv6')(d_uppool3)
        d_conv5= Conv2DTranspose(filters= flags.filters4, kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv5')(d_conv6)
        d_uppool2= UpSampling2D(size= flags.pool_size)(d_conv5)
        d_conv4= Conv2DTranspose(filters= flags.filters3,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv4')(d_uppool2)
        d_conv3= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv3')(d_conv4)
        d_uppool1= UpSampling2D(size= flags.pool_size)(d_conv3)
        d_conv2= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv2')(d_uppool1)
        output_tensor= Conv2DTranspose(filters= 1,  kernel_size= (21, 3),
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'output_tensor')(d_conv2)

        self.model= tf.keras.Model(inputs= input_tensor, outputs= output_tensor)
        self.optimizer= tf.keras.optimizers.Adam(flags.learning_rate1)
        self.loss_object= tf.keras.losses.MeanSquaredError()

class conv8_fc1:
    def __init__(self, flags, input_shape):
        self.flags= flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.build_model()

    def build_model(self):
        flags= self.flags
        self.l2_regul= tf.keras.regularizers.l2(l= flags.l2_regul)

        input_tensor= tf.keras.Input(shape= self.input_shape)
        x= Conv2D(filters= flags.filters1,   kernel_size= (self.num_rows, 3),
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv1')(input_tensor)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters2,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= flags.filters3,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv3')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters4,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= flags.filters5,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv5')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters6,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv6')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= flags.filters7,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul,
                        padding= flags.conv_padding, name= 'e_conv7')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters8,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul,
                        padding= flags.conv_padding, name= 'e_conv8')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        e_pool4= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        e_flatten= Flatten()(e_pool4)

        embedding= Dense(units= flags.embedding_size,  activation= None, name= 'e_fc1')(e_flatten)
        embedding= l2_normalize(embedding)
        self.embedding= embedding
        
        d_flatten= Dense(units= e_flatten.shape[1],    activation= None, name= 'd_fc1')(embedding)
        reshaped= Reshape(target_shape= e_pool4.shape[1:])(d_flatten)
        d_uppool4= UpSampling2D(size= flags.pool_size)(reshaped)
        x= Conv2DTranspose(filters= flags.filters7,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul,
                                 padding= flags.conv_padding, name= 'd_conv8')(d_uppool4)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= flags.filters6,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul,
                                 padding= flags.conv_padding, name= 'd_conv7')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= flags.filters5,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv6')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= flags.filters4, kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv5')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= flags.filters3,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv3')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= flags.filters2,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        output_tensor= Conv2DTranspose(filters= 1,  kernel_size= (21, 3),
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'output_tensor')(x)

        self.model= tf.keras.Model(inputs= input_tensor, outputs= output_tensor)
        self.optimizer= tf.keras.optimizers.Adam(flags.learning_rate1)
        self.loss_object= tf.keras.losses.MeanSquaredError()

class conv12_fc1:
    def __init__(self, flags, input_shape):
        self.flags= flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.filters1= 64
        self.filters2= 64
        self.filters3= 64
        self.filters4= 64
        self.filters5= 128
        self.filters6= 128
        self.filters7= 128
        self.filters8= 128
        self.filters9= 256
        self.filters10= 256
        self.filters11= 256
        self.filters12= 256
        self.build_model()

    def build_model(self):
        flags= self.flags
        self.l2_regul= tf.keras.regularizers.l2(l= flags.l2_regul)

        input_tensor= tf.keras.Input(shape= self.input_shape)
        x= Conv2D(filters= self.filters1,   kernel_size= flags.first_kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv1')(input_tensor)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= self.filters2,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= self.filters3,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv3')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= self.filters4,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= self.filters5,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv5')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= self.filters6,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul, 
                        padding= flags.conv_padding, name= 'e_conv6')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= self.filters7,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul,
                        padding= flags.conv_padding, name= 'e_conv7')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= self.filters8,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul,
                        padding= flags.conv_padding, name= 'e_conv8')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        x= Conv2D(filters= self.filters9,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul,
                        padding= flags.conv_padding, name= 'e_conv9')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= self.filters10,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul,
                        padding= flags.conv_padding, name= 'e_conv10')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= self.filters11,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul,
                        padding= flags.conv_padding, name= 'e_conv11')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= self.filters12,   kernel_size= flags.kernel_size,
                        activation= flags.act_fn1, dtype= tf.float32,
                        kernel_regularizer= self.l2_regul,
                        padding= flags.conv_padding, name= 'e_conv12')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        last_pool= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)
        e_flatten= Flatten()(last_pool)

        embedding= Dense(units= flags.embedding_size,  activation= None, name= 'e_fc1')(e_flatten)
        embedding= l2_normalize(embedding)
        self.embedding= embedding
        
        d_flatten= Dense(units= e_flatten.shape[1],    activation= None, name= 'd_fc1')(embedding)
        reshaped= Reshape(target_shape= last_pool.shape[1:])(d_flatten)
        first_uppool= UpSampling2D(size= flags.pool_size)(reshaped)
        x= Conv2DTranspose(filters= self.filters12,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul,
                                 padding= flags.conv_padding, name= 'd_conv12')(first_uppool)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= self.filters11,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul,
                                 padding= flags.conv_padding, name= 'd_conv11')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= self.filters10,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul,
                                 padding= flags.conv_padding, name= 'd_conv10')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= self.filters9,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul,
                                 padding= flags.conv_padding, name= 'd_conv9')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= self.filters8,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul,
                                 padding= flags.conv_padding, name= 'd_conv8')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= self.filters7,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul,
                                 padding= flags.conv_padding, name= 'd_conv7')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= self.filters6,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv6')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= self.filters5, kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv5')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= self.filters4,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv4')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2DTranspose(filters= self.filters3,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv3')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= UpSampling2D(size= flags.pool_size)(x)
        x= Conv2DTranspose(filters= self.filters2,  kernel_size= flags.kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'd_conv2')(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        output_tensor= Conv2DTranspose(filters= 1,  kernel_size= flags.first_kernel_size,
                                 activation= flags.act_fn1, dtype= tf.float32,
                                 kernel_regularizer= self.l2_regul, 
                                 padding= flags.conv_padding, name= 'output_tensor')(x)

        self.model= tf.keras.Model(inputs= input_tensor, outputs= output_tensor)
        self.optimizer= tf.keras.optimizers.Adam(flags.learning_rate1)
        self.loss_object= tf.keras.losses.MeanSquaredError()
