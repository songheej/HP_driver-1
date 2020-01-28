import tensorflow as tf
from utils.preprocessing.factory_module import factory_class
import os
import pdb
import numpy as np
import importlib

class train_class(factory_class):
    def __init__(self, sys_flags, data_flags, train_flags):
        super(train_class, self).__init__(sys_flags, data_flags, train_flags)
        self.sys_flags= sys_flags
        self.data_flags= data_flags
        self.train_flags= train_flags
        self.make_dirs()
        self.logging()
        self.initialization()
        self.get_model()
        self.train_model()

    def make_dirs(self):
        if not os.path.exists(self.train_flags.dir_results):
            os.makedirs(self.train_flags.dir_results)
        result_list= os.listdir(self.train_flags.dir_results)
        #self.train_name= 'exp_%d'%len(result_list)
        self.train_name= 'exp_1'
        dir_result= os.path.join(self.train_flags.dir_results, self.train_name)
        if not os.path.exists(dir_result):
            os.mkdir(dir_result)
        self.dir_result= dir_result
        self.dir_save_model= os.path.join(dir_result, 'model')
        self.dir_save_log= os.path.join(dir_result, 'log.txt')
        self.dir_save_tfboard= os.path.join(dir_result, 'tfboard')
        if not os.path.exists(self.dir_save_model):
            os.mkdir(self.dir_save_model)
        if not os.path.exists(self.dir_save_tfboard):
            os.mkdir(self.dir_save_tfboard)

    def logging(self):
        print('[i]  Logging Module.')
        self.log= open(self.dir_save_log, 'w')
        self.log.write('-'*10+ self.train_name+ '-'*10+'\n')
        self.log.write('[i]  data flags\n')
        for data_flag in vars(self.data_flags):
            value= getattr(self.data_flags, data_flag)
            if (type(value)== int) or (type(value)== float) or (type(value)== bool):
                line= '      '+ data_flag+ ':'+ str(value)
            elif type(value)== str:
                line= '      '+ data_flag+ ':'+ value
            elif type(value)== list:
                line= '      '+ data_flag+ ':'
                for idx in range(len(value)):
                    line+= value[idx]+', '
            elif type(value)== dict:
                line= '      '+ data_flag+ ':'
                for key in value.keys():
                    line+= key+ '='+ str(value[key])+ ', '
            else:
                pdb.set_trace()
            line+= '\n'
            self.log.write(line)
        self.log.write('[i]  train flags\n')
        for train_flag in vars(self.train_flags):
            value= getattr(self.train_flags, train_flag)
            if (type(value)== int) or (type(value)== float) or (type(value)== bool):
                line= '      '+ train_flag+ ':'+ str(value)
            elif type(value)== str:
                line= '      '+ train_flag+ ':'+ value
            elif type(value)== list:
                line= '      '+ train_flag+ ':'
                for idx in range(len(value)):
                    line+= value[idx]+', '
            elif type(value)== dict:
                line= '      '+ train_flag+ ':'
                for key in value.keys():
                    line+= key+ '='+ str(value[key])+ ', '
            elif type(value)== tuple:
                line= '      '+ train_flag+ ':('
                for idx in range(len(value)):
                    line+= str(value[idx])
                line+= ')'
            else:
                pdb.set_trace()
            line+= '\n'
            self.log.write(line)

    def initialization(self):
        sample_img= np.load(self.train_data[0,0])
        self.num_rows= sample_img.shape[0]
        self.num_train_img= self.train_data.shape[0]
        self.num_test_img= self.test_data.shape[0]
        self.train_indices= np.arange(self.num_train_img)
        self.test_indices= np.arange(self.num_test_img)
        self.train_batch_count= 0
        self.test_batch_count= 0
        self.epoch= 1
        self.iter= 0
        self.test_epoch= 1
        print('[i]  # of training imgs :', self.num_train_img)
        print('[i]  # of testing  imgs :', self.num_test_img)

    def get_model(self):
        input_shape= (self.num_rows, self.data_flags.Ls, 1)

        module_model= importlib.import_module(self.sys_flags.module_model)
        if self.train_flags.model_name== 'conv4_fc2':
            self.model_class= module_model.conv4_fc2(self.train_flags, input_shape)
        elif self.train_flags.model_name== 'conv8_fc2':
            self.model_class= module_model.conv8_fc2(self.train_flags, input_shape)
        elif self.train_flags.model_name== 'conv12_fc2':
            self.model_class= module_model.conv12_fc2(self.train_flags, input_shape)
        else:
            raise ValueError('Model name {} not understood.'.format(self.train_flags.model_name))

        self.model= self.model_class.model
        self.optimizer= tf.keras.optimizers.Adam(learning_rate= self.train_flags.learning_rate)
        self.loss_object= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True)
        self.train_acc= tf.keras.metrics.SparseCategoricalAccuracy(name= 'train_acc')
        self.test_acc= tf.keras.metrics.SparseCategoricalAccuracy(name= 'test_acc')

    def train_next_batch(self, batch_size):
        self.max_batch_count= int(self.num_train_img/ self.train_flags.batch_size)
        if self.train_batch_count== 0:
            np.random.shuffle(self.train_indices)
        batch_img= np.zeros(shape= (batch_size, self.num_rows, self.data_flags.Ls), 
                            dtype= np.float32)
        batch_idx= self.train_indices[self.train_batch_count*self.train_flags.batch_size:\
                            (self.train_batch_count+1)*self.train_flags.batch_size]
        batch_data= self.train_data[batch_idx]
        for i in range(self.train_flags.batch_size):
            # Moving AVG 적용
            #if self.train_flags.use_ma:
            #    Ls= self.data_flags.Ls
            #    side_length= int((Ls+1)/2)- 1
            #    img= np.load(batch_data[i, 0])
            #    template= np.zeros(shape= np.shape(img))
            #    tmp_img[:, 0:side_length]= img[:, 0:side_length]
            #    for k in range(Ls- side_length):
            #        idx= k+ side_length
            #        moving_avg= np.mean(loaded_img[:, idx:idx+flags.moving_avg_window], axis=1)
            #        tmp_img[:, idx]= moving_avg
            #    batch_img[i, :, :]= tmp_img
            batch_img[i, :, :]= np.load(batch_data[i, 0])
        batch_img= np.expand_dims(batch_img, axis=3)
        batch_label= self.train_data[batch_idx, 1].astype(np.int)

        self.train_batch_count+=1
        if self.train_batch_count== self.max_batch_count- 1:
            self.train_batch_count= 0
            self.epoch+= 1
        return batch_img, batch_label

    def test_next_batch(self, batch_size):
        max_batch_count= int(self.num_test_img/ batch_size)
        if self.test_batch_count== 0:
            np.random.shuffle(self.test_indices)
        batch_img= np.zeros(shape= (batch_size, self.num_rows, self.data_flags.Ls), 
                            dtype= np.float32)
        batch_idx= self.test_indices[self.test_batch_count* batch_size:\
                                     (self.test_batch_count+1)* batch_size]
        batch_data= self.test_data[batch_idx]
        for i in range(batch_size):
            #if flags.use_moving_avg:
            #    time_length= np.shape(batch_img)[2]
            #    side_length= int((time_length+1)/2)- 1
            #    loaded_img= np.load(batch_info[i, 1])
            #    tmp_img= np.zeros(shape= np.shape(loaded_img))
            #    tmp_img[:, 0:side_length]= loaded_img[:, 0:side_length]
            #    for k in range(time_length- side_length):
            #        idx= k+ side_length
            #        moving_avg= np.mean(loaded_img[:, idx: idx+ flags.moving_avg_window], axis=1)
            #        tmp_img[:, idx]= moving_avg
            #    batch_img[i, :, :]= tmp_img
            batch_img[i, :, :]= np.load(batch_data[i, 0])
        batch_img= np.expand_dims(batch_img, axis=3)
        batch_label= self.test_data[batch_idx, 1].astype(np.int)

        self.test_batch_count+=1
        if self.test_batch_count== max_batch_count- 1:
            self.test_batch_count= 0
            self.test_epoch+= 1
        return batch_img, batch_label

    def train_model(self):
        train_writer= tf.summary.create_file_writer(self.dir_save_tfboard+ '/train_summary')
        test_writer= tf.summary.create_file_writer(self.dir_save_tfboard+ '/test_summary')
        weights_writer= tf.summary.create_file_writer(self.dir_save_tfboard+ '/weight_summary')
        ckpt= os.path.join(self.dir_save_model, 'cp-{acc:04f}.ckpt')
        dir_ckpt= os.path.dirname(ckpt)
        dir_iterations= self.dir_result+ '/iteraions.npy'
        if os.path.exists(dir_iterations):
            self.iter= np.load(dir_iterations)
            self.max_batch_count= int(self.num_train_img/ self.train_flags.batch_size)
            self.epoch= int(self.iter/ self.max_batch_count)+ 1
            latest_ckpt= tf.train.latest_checkpoint(dir_ckpt)
            self.model.load_weights(latest_ckpt)
            print('   ')
            print('[i]  Continue training from %d to %d epochs'%(self.epoch, self.train_flags.num_epoch))
            print('[i]  Iteration starts from %d'%(self.iter))
            print('   ')
        max_test_acc= 0
        max_train_acc= 0
        while(self.epoch< self.train_flags.num_epoch+ 1):
            batch_img, batch_label= self.train_next_batch(self.train_flags.batch_size)
            with tf.GradientTape() as tape:
                logits= self.model(batch_img)
                softmax= tf.nn.softmax(logits)
                train_loss_value= self.loss_object(batch_label, softmax)
            train_acc= self.train_acc(batch_label, logits)* 100
            grads= tape.gradient(train_loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            with train_writer.as_default():
                tf.summary.scalar('loss', train_loss_value, step= self.iter)
                tf.summary.scalar('acc', train_acc, step= self.iter)
            train_writer.flush()
            self.train_acc.reset_states()
            self.iter+= 1

            if self.iter% 100== 0:
                np.save(dir_iterations, self.iter)
                batch_img, batch_label= self.test_next_batch(self.train_flags.test_batch_size)
                logits= self.model(batch_img)
                softmax= tf.nn.softmax(logits)
                test_loss_value= self.loss_object(batch_label, softmax)
                test_acc= self.test_acc(batch_label, logits)* 100
                with test_writer.as_default():
                    tf.summary.scalar('loss', test_loss_value, step= self.iter)
                    tf.summary.scalar('acc', test_acc, step= self.iter)
                    test_writer.flush()
                    self.test_acc.reset_states()
                line= 'iter : %d,  train acc : %.2f,  test acc : %.2f\n'\
                        %(self.iter, train_acc, test_acc)
                self.log.write(line)
                line= '            max train acc : %.2f,  max test acc : %.2f\n'\
                        %(max_train_acc, max_test_acc)
                self.log.write(line)
                weights_writer.flush()

                #### Overfitting prevented save
                if max_test_acc< test_acc:
                    max_test_acc= test_acc
                    dir_ckpt= os.path.dirname(ckpt)
                    self.model.save_weights(ckpt.format(acc= int(max_test_acc)))

                    print('model saved')
                if max_train_acc< train_acc:
                    max_train_acc= train_acc
                print('max train acc : %.2f, max test acc : %.2f'\
                        %(max_train_acc, max_test_acc))

