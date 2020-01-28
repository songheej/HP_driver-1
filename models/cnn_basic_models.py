import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, \
                                    LayerNormalization, BatchNormalization, Activation,\
                                    Add
import pdb

class conv4_fc2:
    def __init__(self, train_flags, input_shape):
        self.train_flags= train_flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.build_model()

    def build_model(self):
        l2_regul= tf.keras.regularizers.l2(l= self.train_flags.l2_regul)
        self.layer_names= ['conv1', 'conv2', 'conv3', 'conv4']

        input_tensor= tf.keras.Input(shape= self.input_shape)
        x= Conv2D(filters= self.train_flags.filters1,  kernel_size= self.train_flags.kernel_size1,
                  activation= self.train_flags.act_fn,  dtype= tf.float32,
                  kernel_regularizer= l2_regul,
                  strides= self.train_flags.conv_strides,
                  padding= 'valid',  name= self.layer_names[0])(input_tensor)
        x= Conv2D(filters= self.train_flags.filters2,  kernel_size= self.train_flags.kernel_size2,
                  activation= self.train_flags.act_fn,  dtype= tf.float32,
                  kernel_regularizer= l2_regul,
                  strides= self.train_flags.conv_strides,
                  padding= self.train_flags.conv_padding,  name= self.layer_names[1])(x)
        x= MaxPool2D(pool_size= self.train_flags.pool_size,  
                     padding= self.train_flags.pool_padding)(x)

        x= Conv2D(filters= self.train_flags.filters2,  kernel_size= self.train_flags.kernel_size3,
                  activation= self.train_flags.act_fn,  dtype= tf.float32,
                  kernel_regularizer= l2_regul,
                  strides= self.train_flags.conv_strides,
                  padding= self.train_flags.conv_padding,  name= self.layer_names[2])(x)
        x= Conv2D(filters= self.train_flags.filters2, kernel_size= self.train_flags.kernel_size4,
                  activation= self.train_flags.act_fn,  dtype= tf.float32,
                  kernel_regularizer= l2_regul,
                  strides= self.train_flags.conv_strides,
                  padding= self.train_flags.conv_padding,  name= self.layer_names[3])(x)

        x= Flatten()(x)
        x= Dense(units= self.train_flags.fc_unit1, activation= self.train_flags.act_fn)(x)
        logits= Dense(units= self.train_flags.num_class,  activation= None)(x)
        self.model= tf.keras.Model(inputs= input_tensor, outputs= logits)

class conv8_fc2:
    def __init__(self, flags, input_shape):
        self.flags= flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.build_model()

    def build_model(self):
        flags= self.flags
        self.l2_regul= tf.keras.regularizers.l2(l= flags.l2_regul)
        self.layer_names= ['conv1', 'conv2', 'conv3', 'conv4', 
                           'conv5', 'conv6', 'conv7', 'conv8']

        input_tensor= tf.keras.Input(shape= self.input_shape)
        x= Conv2D(filters= flags.filters1,  kernel_size= (self.num_rows, 5),
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= 'valid',  name= self.layer_names[0])(input_tensor)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters1,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[1])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size,  padding= flags.pool_padding)(x)

        x= Conv2D(filters= flags.filters2,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[2])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters2,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[3])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size,  padding= flags.pool_padding)(x)

        x= Conv2D(filters= flags.filters2,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[4])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters2, kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[5])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)

        x= Conv2D(filters= flags.filters3,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[6])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters3, kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[7])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size,  padding= flags.pool_padding)(x)

        x= Flatten()(x)
        x= Dense(units= flags.units1, activation= flags.fc_act_fn1)(x)
        logits= Dense(units= flags.num_classes,  activation= flags.last_act_fn)(x)
        self.model= tf.keras.Model(inputs= input_tensor, outputs= logits)


class conv12_fc2:
    def __init__(self, flags, input_shape):
        self.flags= flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.build_model()

    def build_model(self):
        flags= self.flags
        self.l2_regul= tf.keras.regularizers.l2(l= flags.l2_regul)
        self.layer_names= ['conv1', 'conv2', 'conv3', 'conv4', 
                           'conv5', 'conv6', 'conv7', 'conv8',
                           'conv9', 'conv10', 'conv11', 'conv12']

        input_tensor= tf.keras.Input(shape= self.input_shape)
        x= Conv2D(filters= flags.filters1,  kernel_size= (self.num_rows, 5),
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= 'valid',  name= self.layer_names[0])(input_tensor)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters1,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[1])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size,  padding= flags.pool_padding)(x)

        x= Conv2D(filters= flags.filters2,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[2])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters2,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[3])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size,  padding= flags.pool_padding)(x)

        x= Conv2D(filters= flags.filters2,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[4])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters2, kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[5])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size, padding= flags.pool_padding)(x)

        x= Conv2D(filters= flags.filters3,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[6])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters3, kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[7])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= MaxPool2D(pool_size= flags.pool_size,  padding= flags.pool_padding)(x)

        x= Conv2D(filters= flags.filters3,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[8])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters3, kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[9])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)

        x= Conv2D(filters= flags.filters4,  kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[10])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)
        x= Conv2D(filters= flags.filters4, kernel_size= flags.kernel_size,
                  activation= flags.act_fn1,  dtype= tf.float32,
                  kernel_regularizer= self.l2_regul,
                  strides= flags.conv_strides,
                  padding= flags.conv_padding,  name= self.layer_names[11])(x)
        if flags.use_batchnorm:
            x= BatchNormalization()(x)

        x= MaxPool2D(pool_size= flags.pool_size,  padding= flags.pool_padding)(x)
        x= MaxPool2D(pool_size= flags.pool_size,  padding= flags.pool_padding)(x)
        x= Flatten()(x)
        x= Dense(units= flags.units1, activation= flags.fc_act_fn1, name= 'fc1')(x)
        x= Dense(units= flags.units2, activation= flags.fc_act_fn1)(x)
        logits= Dense(units= flags.num_classes,  activation= flags.last_act_fn, name='fc2')(x)
        self.model= tf.keras.Model(inputs= input_tensor, outputs= logits)

class renset50:
    def __init__(self, flags, input_shape):
        self.flags= flags
        self.input_shape= input_shape
        self.num_rows= input_shape[0]
        self.build_model()

    def build_model(self):
        flags= self.flags
        input_tensor= tf.keras.Input(shape= self.input_shape)


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides= (1,2)):
        flags= self.flags
        filters1, filters2, filters3= filters
        cnn_name_base= 'res'+ str(stage)+ block+ '_branch'
        bn_name_base= 'bn'+ str(stage)+ block+ '_branch'
        x= Conv2D(filters= filters1, kernel_size= kernel_size, dtype= tf.float32,
                  strides= strides, padding= flags.conv_padding, name= cnn_base_name+ '2a')(input_tensor)
        x= BatchNormalization(name= bn_name_base+ '2a')(x)
        x= Activation('relu')(x)

        x= Conv2D(filters= filters2, kernel_size= kernel_size, dtype= tf.float32,
                  strides= strides, padding= flags.conv_padding, name= cnn_base_name+ '2b')(x)
        x= BatchNormalization(name= bn_name_base+ '2b')(x)
        x= Activation('relu')(x)

        x= Conv2D(filters= filters3, kernel_size= kernel_size, dtype= tf.float32,
                  strides= strides, padding= flags.conv_padding, name= cnn_base_name+ '2c')(x)
        x= BatchNormalization(name= bn_name_base+ '2c')(x)

        shortcut= Conv2D(filters= filters3, kernel_size= kernel_size, dtype= tf.float32,
                         strides= stides, padding= flags.conv_padding, 
                         kernel_initializer= 'he_normal', name= conv_base_name+ '1')(input_tensor)
        shortcut= BatchNormalization(name= bn_name_base+ '1')(shortcut)
        x= Add([x, shortcut])
        x= Activation('relu')(x)
        return x
