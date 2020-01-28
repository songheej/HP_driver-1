import tensorflow as tf
from utils.preprocessing.factory_module import factory_class
import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import importlib

class test_class(factory_class):
    def __init__(self, sys_flags, data_flags, train_flags):
        super(test_class, self).__init__(sys_flags, data_flags, train_flags)
        self.sys_flags= sys_flags
        self.data_flags= data_flags
        self.train_flags= train_flags

        self.initialization()
        self.get_data_label()
        self.get_model()

        self.class_accuracy()

    def initialization(self):
        sample_img= np.load(self.train_data[0, 0])
        self.num_rows= sample_img.shape[0]
        self.train_name= 'exp_1'
        self.dir_result= os.path.join(self.train_flags.dir_results, self.train_name)
        self.dir_save_model= os.path.join(self.dir_result, 'model')
        dir_label= os.path.join(self.dir_processed_data, 'labels')
        self.train_s_names= np.load(os.path.join(dir_label, 'train_s.npy'))
        self.train_m_names= np.load(os.path.join(dir_label, 'train_m.npy'))
        self.test_s_names= np.load(os.path.join(dir_label, 'test_s.npy'))
        self.test_m_names= np.load(os.path.join(dir_label, 'test_m.npy'))
        print('[i]  Train, Test label Loaded')

    def get_data_label(self):
        self.data_all= {}
        self.label_all= {}
        for driver_name in self.driver_names:
            dir_img= os.path.join(self.dir_imgs, driver_name)
            img_list= os.listdir(dir_img)
            self.data_all[driver_name]= []
            for img_name in img_list:
                dir_img= os.path.join(dir_img, img_name)
                self.data_all[driver_name].append(dir_img)
            if (driver_name in self.train_s_names) or (driver_name in self.test_s_names):
                label= '1'
            elif (driver_name in self.train_m_names) or (driver_name in self.test_m_names):
                label= '0'
            else:
                label= '-1'
            self.label_all[driver_name]= label

    def get_model(self):
        input_shape= (self.num_rows, self.data_flags.Ls, 1)
        module_model= importlib.import_module(self.sys_flags.module_model)
        self.model_class= module_model.conv4_fc2(self.train_flags, input_shape)
        self.model= self.model_class.model
        self.model.load_weights(tf.train.latest_checkpoint(self.dir_save_model))
        self.loss_object= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True)
        self.accuracy= tf.keras.metrics.SparseCategoricalAccuracy(name= 'Accuracy')

    def class_accuracy(self):
        # sporty, mild의 accuracy를 각각 비교
        data_dict= {}
        def get_accuracy(input_dict):
            for data_name in input_dict.keys():
                data= input_dict[data_name]
                num_data= data.shape[0]
                batch_size= 256
                num_batch= num_data// batch_size
                for batch_iter in range(num_batch):
                    batch_idx= np.arange(batch_iter* batch_size, (batch_iter+ 1)* batch_size)
                    batch_data= data[batch_idx]
                    img_loaded= np.zeros(shape= (batch_size, self.num_rows, self.data_flags.Ls),
                                         dtype= np.float32)
                    label_loaded= data[batch_idx, 1].astype(np.int)
                    for img_idx in range(batch_size):
                        img_loaded[img_idx]= np.load(batch_data[img_idx, 0])
                    img_loaded= np.expand_dims(img_loaded, axis=3)
                    logits= self.model(img_loaded)
                    acc= self.accuracy(label_loaded, logits)*100
                    print(acc)
                    if batch_iter== 10:
                        pdb.set_trace()

        train_s_indices= np.where(self.train_data[:, 1]== '1')[0]
        train_m_indices= np.where(self.train_data[:, 1]== '0')[0]
        test_s_indices= np.where(self.test_data[:, 1]== '1')[0]
        test_m_indices= np.where(self.test_data[:, 1]== '0')[0]
        train_s_data= self.train_data[train_s_indices, :]
        train_m_data= self.train_data[train_m_indices, :]
        test_s_data= self.test_data[test_s_indices, :]
        test_m_data= self.test_data[test_m_indices, :]
        data_dict['train_s']= train_s_data
        data_dict['train_m']= train_m_data
        data_dict['test_s']= test_s_data
        data_dict['test_m']= test_m_data
        get_accuracy(data_dict)


        pdb.set_trace()
