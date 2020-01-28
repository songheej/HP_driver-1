import os
from utils.preprocessing.dir_setting import dir_setting
import pickle
import pdb
import pandas as pd
import numpy as np
import pathlib
import shutil
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io
from time import time
import copy

class factory_class(dir_setting):
    def __init__(self, sys_flags, data_flags, train_flags):
        print('[i]----------Factory Module----------')
        super(factory_class, self).__init__(sys_flags, data_flags, train_flags)
        self.sys_flags= sys_flags
        self.data_flags= data_flags
        self.train_flags= train_flags
        self.initializations()
        self.get_statistical_value()
        for driver_idx, driver_name in enumerate(self.driver_names):
            necessity= self.processing_necessity(driver_name)
            if necessity== True:
                print('[i]  Preprocessing in progress [%d/%d]'%(driver_idx, self.num_drivers))
                self.signal_selection(driver_name)
                self.whl_spd_avg(driver_name)
                self.dict_to_arr(driver_name)
                self.make_frame(driver_name)
                if self.data_flags.use_normalize== True:
                    self.data_normalize(driver_name)
                self.data_segmentation(driver_name)
        print('[i]  Factoy Module End.')
        self.get_train_test_driver_names()
        self.get_train_test_img_label()
        self.get_avg_spd()
        #self.data_histogram()

    def initializations(self):
        self.dir_data_check= os.path.join(self.dir_processed_data, 'data_check')
        self.dir_statistics= os.path.join(self.dir_processed_data, 'statistics')
        self.dir_imgs= os.path.join(self.dir_processed_data, 'imgs')
        self.dir_label= os.path.join(self.dir_processed_data, 'labels')
        self.dir_mean_values= os.path.join(self.dir_statistics, 'mean_values.npy')
        self.dir_stddev_values= os.path.join(self.dir_statistics, 'stddev_values.npy')
        self.dir_processing_necessity= os.path.join(self.dir_processed_data, 'processing_necessity.pkl')
        self.dir_frames= os.path.join(self.dir_processed_data, 'frames')
        self.dir_avg_spd= os.path.join(self.dir_imgs, 'avg_spd.npy')

        if not os.path.exists(self.dir_data_check):
            os.mkdir(self.dir_data_check)
        if not os.path.exists(self.dir_frames):
            os.mkdir(self.dir_frames)
        if not os.path.exists(self.dir_statistics):
            os.mkdir(self.dir_statistics)
        if not os.path.exists(self.dir_imgs):
            os.mkdir(self.dir_imgs)
        if not os.path.exists(self.dir_label):
            os.mkdir(self.dir_label)

    def processing_necessity(self, driver_name):
        if not os.path.exists(self.dir_processing_necessity):
            with open(self.dir_processing_necessity, 'wb') as f:
                pickle.dump({driver_name:True}, f, pickle.HIGHEST_PROTOCOL)
            return True
        else:
            with open(self.dir_processing_necessity, 'rb') as f:
                necessities= pickle.load(f)
            if driver_name in necessities.keys():
                return necessities[driver_name]
            else:
                necessities[driver_name]= True
            return True

    def signal_selection(self, data_name):
        mat_name= 'Recorder_2019-'+ data_name
        dir_mat= os.path.join(self.sys_flags.dir_mat, mat_name)
        mat_data= scipy.io.loadmat(dir_mat)
        mat_keys= mat_data.keys()
        self.t_names= []
        self.t_data= {}
        self.s_names= []
        self.s_data= {}
        for key in mat_keys:
            key_name_refined= '_'.join(key.split('_')[0:-1])
            t_name= key.split('_')[-1]
            if key_name_refined in self.data_flags.signal_names:
                self.s_names.append(key)
                self.s_data[key]= mat_data[key]
                if t_name not in self.t_names:
                    self.t_names.append(t_name)
                    self.t_data[t_name]= mat_data[t_name]

    def whl_spd_avg(self, driver_name):
        s_name_delete= []
        for s_name in self.s_names:
            ts_name= s_name.split('_')[-1]
            if '_'.join(s_name.split('_')[0:-1])== 'WHL_SPD_RR':
                s_name_delete.append(s_name)
                RR_data= self.s_data[s_name]
            if '_'.join(s_name.split('_')[0:-1])== 'WHL_SPD_RL':
                s_name_delete.append(s_name)
                RL_data= self.s_data[s_name]
            if '_'.join(s_name.split('_')[0:-1])== 'WHL_SPD_FR':
                s_name_delete.append(s_name)
                FR_data= self.s_data[s_name]
            if '_'.join(s_name.split('_')[0:-1])== 'WHL_SPD_FL':
                s_name_delete.append(s_name)
                FL_data= self.s_data[s_name]
        avg_data= (RR_data+ RL_data+ FR_data+ FL_data)/4
        whl_signal_name= 'WHL_SPD_AVG'+ '_'+ ts_name
        self.s_names.append(whl_signal_name)
        self.s_data[whl_signal_name]= avg_data
        for name_delete in s_name_delete:
            del self.s_data[name_delete]
            self.s_names.remove(name_delete)
        for s_name in self.s_names:
            dir_plot= os.path.join(self.dir_data_check, driver_name+'_%s.png'%(s_name))
            if not os.path.exists(dir_plot):
                data= self.s_data[s_name]
                t_arange= np.arange(len(data))
                plt.figure()
                plt.plot(t_arange, data)
                plt.title(s_name)
                plt.savefig(dir_plot)
                plt.close()

    def get_statistical_value(self):
        if not (os.path.exists(self.dir_mean_values) or os.path.exists(self.dir_stddev_values)):
            self.vis_count= 1
            print('[i]  Calculating statistical values of drivers')
            num_s= len(self.data_flags.signal_names)- 3
            mean_values_all= []
            stddev_values_all= []
            #mean_values_all= np.zeros(shape= (self.num_drivers, num_s))
            #stddev_values_all= np.zeros(shape= (self.num_drivers, num_s))
            for driver_idx, driver_name in enumerate(self.driver_names):
                self.signal_selection(driver_name)
                self.whl_spd_avg(driver_name)
                mean_values= [0]* num_s
                stddev_values= [0]* num_s
                for s_name in self.s_names:
                    s_name_refined= '_'.join(s_name.split('_')[:-1])
                    s_idx= self.data_flags.signal_order[s_name_refined]
                    mean_value= np.mean(self.s_data[s_name])
                    stddev_value= np.std(self.s_data[s_name])
                    mean_values[s_idx]= mean_value
                    stddev_values[s_idx]= stddev_value
                mean_values_all.append(mean_values)
                stddev_values_all.append(stddev_values)
                del mean_values
                del stddev_values
                if driver_idx% 10== 0:
                    print('[i]    %d / %d'%(driver_idx, self.num_drivers))
            mean_values_all= np.mean(mean_values_all, axis=0)
            stddev_values_all= np.mean(stddev_values_all, axis=0)
            np.save(self.dir_mean_values, mean_values_all)
            np.save(self.dir_stddev_values, stddev_values_all)

    def dict_to_arr(self, driver_name):
        num_s= len(self.s_names)
        # 가장 긴 time length 계산
        max_t_length= 0
        for s_name in self.s_names:
            t_name= s_name.split('_')[-1]
            t_length= len(self.t_data[t_name])
            if max_t_length< t_length:
                max_t_length= t_length
        s_data= np.zeros(shape= (num_s, max_t_length))
        t_data= np.zeros(shape= (num_s, max_t_length))
        for s_name in self.s_names:
            s_name_refined= '_'.join(s_name.split('_')[:-1])
            s_idx= self.data_flags.signal_order[s_name_refined]
            t_name= s_name.split('_')[-1]
            s_len= len(self.s_data[s_name])
            s_data[s_idx, 0:s_len]= np.squeeze(self.s_data[s_name])
            t_data[s_idx, 0:s_len]= np.squeeze(self.t_data[t_name])
        self.s_data= s_data
        self.t_data= t_data

    def make_frame(self, driver_name):
        print('[i]    Synchronization started.')
        dir_frame_s= os.path.join(self.dir_frames, '%s_signal.npy'%(driver_name))
        dir_frame_t= os.path.join(self.dir_frames, '%s_time.npy'%(driver_name))
        if (not os.path.exists(dir_frame_s)) or (not os.path.exists(dir_frame_t)): 
            self.vis_count= 1
            Ts= self.data_flags.Ts
            num_s= len(self.s_names)
            t_idx_arr= (self.t_data/Ts).astype(np.int32)
            t_size= np.shape(t_idx_arr)[1]
            t_idx_arr= t_idx_arr[:, :t_size]
            max_t_idx= np.max(t_idx_arr)
            t_idx= np.zeros(shape= (num_s, max_t_idx), dtype= np.int32)
            t_now= time()
            # Int time indices 중 중복된 배수들을 제거.
            t_idx= []
            for idx in range(1, max_t_idx):
                int_idxs= np.argmax(t_idx_arr== idx, axis=1)
                t_idx.append(int_idxs)
                if idx%3000 == 0:
                    print('[i]    %d / %d'%(idx, max_t_idx))
            t_idx= np.transpose(np.array(t_idx))
            t_idx= np.delete(t_idx, 0, axis=1)
            t_data= np.zeros(shape=np.shape(t_idx))
            for idx in range(len(self.s_names)):
                t_data[idx, :]= self.t_data[idx, t_idx[idx, :]]
            # Left, Right 값 비교하여 더 가까운 time idx 가져옴
            # 이때, time_data, time_idx 모두 계산
            for s_name in self.s_names:
                s_name_refined= '_'.join(s_name.split('_')[:-1])
                s_idx= self.data_flags.signal_order[s_name_refined]
                t_data_r= self.t_data[s_idx, t_idx[s_idx, :]]
                t_data_l= self.t_data[s_idx, t_idx[s_idx, :]-1 ]
                t_data_lr= np.zeros(shape= (2, len(t_data_r)))
                t_data_lr[0, :]= t_data_l
                t_data_lr[1, :]= t_data_r
                t_delta= t_data_r+ t_data_l- 2*np.arange(1,len(t_data_r)+1)*Ts
                t_delta_bool= np.less(t_delta, 0).astype(np.int8)
                for i in range(len(t_data_r)):
                    if i== 0:
                        t_data[s_idx, i]= t_data_r[i]
                        t_idx[s_idx, i]= t_idx[s_idx, i]
                    else:
                        if t_delta_bool[i]== 0:
                            t_idx[s_idx, i]= t_idx[s_idx, i]- 1
                        elif t_delta_bool[i]== 1:
                            t_idx[s_idx, i]= t_idx[s_idx, i]
                        t_data[s_idx, i]= t_data_lr[t_delta_bool[i], i]
            s_data_sampled= np.zeros(shape=np.shape(t_data))
            for s_name in self.s_names:
                s_name_refined= '_'.join(s_name.split('_')[:-1])
                s_idx= self.data_flags.signal_order[s_name_refined]
                s_data_sampled[s_idx, :]= self.s_data[s_idx, t_idx[s_idx, :]]
            self.t_data= t_data
            self.s_data= s_data_sampled
            np.save(dir_frame_s, self.s_data)
            np.save(dir_frame_t, self.t_data)
        else:
            print('[i]    Frame already exists.')
            self.s_data= np.load(dir_frame_s)
            self.t_data= np.load(dir_frame_t)

    def data_normalize(self, driver_name):
        self.mean_values_all= np.load(self.dir_mean_values)
        self.stddev_values_all= np.load(self.dir_stddev_values)
        mean_values_all= np.expand_dims(self.mean_values_all, axis=1)
        stddev_values_all= np.expand_dims(self.stddev_values_all, axis= 1)
        self.s_data= self.s_data- mean_values_all
        self.s_data= np.divide(self.s_data, stddev_values_all)
        self.s_data= self.s_data* self.data_flags.normalize_value
        for s_name in self.s_names:
            dir_plot= os.path.join(self.dir_data_check, driver_name+'_%s_processed.png'%(s_name))
            if not os.path.exists(dir_plot):
                s_name_refined= '_'.join(s_name.split('_')[:-1])
                s_idx= self.data_flags.signal_order[s_name_refined]
                data= self.s_data[s_idx, :]
                t_arange= np.arange(len(data))
                plt.figure()
                plt.plot(t_arange, data)
                plt.title(s_name)
                plt.savefig(dir_plot)
                plt.close()

    def data_segmentation(self, driver_name):
        print('[i]    Data segmentation started.')
        self.vis_count= 1
        dir_driver_img= os.path.join(self.dir_imgs, driver_name)
        if not os.path.exists(dir_driver_img):
            os.makedirs(dir_driver_img)
        Ls= self.data_flags.Ls
        Ls_shift= self.data_flags.Ls_shift
        num_row, t_length= np.shape(self.s_data)
        seg_num= int((t_length- Ls)/ Ls_shift- 1)
        signal_data_processed= []
        img_idx= 0
        assert seg_num>0, 'data name {} has test seg num less than 0'.format(driver_name)
        
        for seg_idx in range(seg_num):
            img= self.s_data[:, Ls_shift*seg_idx: Ls_shift*seg_idx+Ls]
            img_name= str(img_idx)+ '.npy'
            dir_img= os.path.join(dir_driver_img, img_name)
            np.save(dir_img, img)
            img_idx+= 1
        f= open(self.dir_processing_necessity, 'rb')
        necessities= pickle.load(f)
        necessities[driver_name]= False
        f= open(self.dir_processing_necessity, 'wb')
        pickle.dump(necessities, f, pickle.HIGHEST_PROTOCOL)

    def get_train_test_driver_names(self):
        # train, test 개수만큼 sporty, mild 할당
        print('[i]  Get train, test driver names')
        train_ratio= self.train_flags.train_ratio
        num_train_s= int(len(self.sporty_names)* train_ratio)
        num_train_m= int(len(self.mild_names)* train_ratio)
        num_train_n= int(len(self.mid_names)* train_ratio)
        #num_test_n= int(num_train_n- len(self.))
        np.random.shuffle(self.sporty_names)
        np.random.shuffle(self.mild_names)
        np.random.shuffle(self.mid_names)
        train_s_names, test_s_names= np.split(self.sporty_names, [num_train_s])
        train_m_names, test_m_names= np.split(self.mild_names, [num_train_m])
        train_n_names, test_n_names= np.split(self.mid_names, [num_train_n])
        np.save(os.path.join(self.dir_label, 'train_s.npy'), train_s_names)
        np.save(os.path.join(self.dir_label, 'test_s.npy'), test_s_names)
        np.save(os.path.join(self.dir_label, 'train_m.npy'), train_m_names)
        np.save(os.path.join(self.dir_label, 'test_m.npy'), test_m_names)
        self.train_label= []
        self.test_label= []
        self.normal_train_label= []
        self.normal_test_label= []
        driver_idx_name= {}
        print('[i]    Total %d drivers exists'%(self.num_drivers))
        for driver_int, driver_name in enumerate(self.driver_names):
            driver_idx_name[driver_name]= driver_int
        for name in train_s_names:
            orig_idx= driver_idx_name[name]
            label= int(self.int_labels[orig_idx])
            name_label= [name, label]
            self.train_label.append(name_label)
        for name in train_m_names:
            orig_idx= driver_idx_name[name]
            label= int(self.int_labels[orig_idx])
            name_label= [name, label]
            self.train_label.append(name_label)
        for name in test_s_names:
            orig_idx= driver_idx_name[name]
            label= int(self.int_labels[orig_idx])
            name_label= [name, label]
            self.test_label.append(name_label)
        for name in test_m_names:
            orig_idx= driver_idx_name[name]
            label= int(self.int_labels[orig_idx])
            name_label= [name, label]
            self.test_label.append(name_label)
        for name in train_n_names:
            orig_idx= driver_idx_name[name]
            label= int(self.int_labels[orig_idx])
            name_label= [name, label]
            self.normal_train_label.append(name_label)
        for name in test_n_names:
            orig_idx= driver_idx_name[name]
            label= int(self.int_labels[orig_idx])
            name_label= [name, label]
            self.normal_test_label.append(name_label)
        self.train_label= np.array(self.train_label)
        self.test_label= np.array(self.test_label)
        self.normal_train_label= np.array(self.normal_train_label)
        self.normal_test_label= np.array(self.normal_test_label)
        self.num_train_drivers= self.train_label.shape[0]
        self.num_test_drivers= self.test_label.shape[0]

    def get_train_test_img_label(self):
        # 각 driver_name에 속한 img list를 받아오고, label 지정해줌
        self.train_data= []
        self.test_data= []
        self.normal_train_data= []
        self.normal_test_data= []
        for train_idx in range(self.num_train_drivers):
            [name, label]= self.train_label[train_idx]
            dir_imgs= os.path.join(self.dir_imgs, name)
            img_list= os.listdir(dir_imgs)
            for img_name in img_list:
                dir_img= os.path.join(dir_imgs, img_name)
                self.train_data.append([dir_img, label])
        for test_idx in range(self.num_test_drivers):
            [name, label]= self.test_label[test_idx]
            dir_imgs= os.path.join(self.dir_imgs, name)
            img_list= os.listdir(dir_imgs)
            for img_name in img_list:
                dir_img= os.path.join(dir_imgs, img_name)
                self.test_data.append([dir_img, label])
        for normal_train_idx in range(self.normal_train_label.shape[0]):
            [name, label]= self.normal_train_label[normal_train_idx]
            dir_imgs= os.path.join(self.dir_imgs, name)
            img_list= os.listdir(dir_imgs)
            for img_name in img_list:
                dir_img= os.path.join(dir_imgs, img_name)
                self.normal_train_data.append([dir_img, label])
        for normal_test_idx in range(self.normal_test_label.shape[0]):
            [name, label]= self.normal_test_label[normal_test_idx]
            dir_imgs= os.path.join(self.dir_imgs, name)
            img_list= os.listdir(dir_imgs)
            for img_name in img_list:
                dir_img= os.path.join(dir_imgs, img_name)
                self.normal_test_data.append([dir_img, label])
        self.train_data= np.array(self.train_data)
        self.test_data= np.array(self.test_data)
        self.normal_train_data= np.array(self.normal_train_data)
        self.normal_test_data= np.array(self.normal_test_data)
        self.num_train_data= self.train_data.shape[0]
        self.num_test_data= self.test_data.shape[0]

    def get_avg_spd(self):
        print('[i]  Speed Thresholding')
        dir_avg= os.path.join(self.dir_processed_data, 'avg_spd')
        if not os.path.exists(dir_avg):
            os.mkdir(dir_avg)
        dir_train= os.path.join(dir_avg, 'train.npy')
        dir_delete_train= os.path.join(dir_avg, 'delete_train.npy')
        dir_test= os.path.join(dir_avg, 'test.npy')
        dir_delete_test= os.path.join(dir_avg, 'delete_test.npy')
        dir_normal= os.path.join(dir_avg, 'normal.npy')
        if (not os.path.exists(dir_train)) or\
           (not os.path.exists(dir_delete_train)) or\
           (not os.path.exists(dir_test)) or\
           (not os.path.exists(dir_delete_test)):
            avg_spd_idx= self.data_flags.signal_order['WHL_SPD_AVG']
            mean_values_all= np.load(self.dir_mean_values)
            stddev_values_all= np.load(self.dir_stddev_values)
            mean_spd= mean_values_all[avg_spd_idx]
            std_spd= stddev_values_all[avg_spd_idx]
            norm_spd_threshold= (self.train_flags.spd_threshold- mean_spd)/ std_spd
            train_avg_spd= []
            delete_train_avg_spd= []
            test_avg_spd= []
            delete_test_avg_spd= []
            test_avg_spd= []
            normal_avg_spd= []
            for train_idx in range(self.num_train_data):
                img= np.load(self.train_data[train_idx, 0])
                img_avg_spd= np.mean(img[avg_spd_idx, :])
                data_arr= []
                data_arr.append(self.train_data[train_idx, 0])
                data_arr.append(self.train_data[train_idx, 1])
                data_arr.append(img_avg_spd)
                if img_avg_spd> norm_spd_threshold:
                    train_avg_spd.append(data_arr)
                else:
                    delete_train_avg_spd.append(data_arr)
                if train_idx%10000== 0:
                    print('    ', train_idx,'/', self.num_train_data)
            for test_idx in range(self.num_test_data):
                img= np.load(self.test_data[test_idx, 0])
                img_avg_spd= np.mean(img[avg_spd_idx, :])
                data_arr= []
                data_arr.append(self.train_data[train_idx, 0])
                data_arr.append(self.train_data[train_idx, 1])
                data_arr.append(img_avg_spd)
                if img_avg_spd> norm_spd_threshold:
                    test_avg_spd.append(data_arr)
                else:
                    delete_test_avg_spd.append(data_arr)
                if test_idx% 10000== 0:
                    print('    ', test_idx, '/', self.num_test_data)
            for idx in range(self.normal_train_data.shape[0]):
                img= np.load(self.normal_train_data[idx, 0])
                img_avg_spd= np.mean(img[avg_spd_idx, :])
                data_arr= []
                data_arr.append(self.normal_train_data[idx, 0])
                data_arr.append(self.normal_train_data[idx, 1])
                data_arr.append(img_avg_spd)
                if img_avg_spd> norm_spd_threshold:
                    normal_avg_spd.append(data_arr)
                if idx% 10000== 0:
                    print('    ', idx, '/', self.normal_train_data.shape[0])
            train_avg_spd= np.array(train_avg_spd)
            delete_train_avg_spd= np.array(delete_train_avg_spd)
            test_avg_spd= np.array(test_avg_spd)
            delete_test_avg_spd= np.array(delete_test_avg_spd)
            normal_avg_spd= np.array(normal_avg_spd)
            np.save(dir_train, train_avg_spd)
            np.save(dir_delete_train, delete_train_avg_spd)
            np.save(dir_test, test_avg_spd)
            np.save(dir_delete_test, delete_test_avg_spd)
            np.save(dir_normal, normal_avg_spd)
        else:
            train_avg_spd= np.load(dir_train)
            delete_train_avg_spd= np.load(dir_delete_train)
            test_avg_spd= np.load(dir_test)
            delete_test_avg_spd= np.load(dir_delete_test)
        train_delete_ratio= delete_train_avg_spd.shape[0]/ train_avg_spd.shape[0]* 100
        test_delete_ratio= delete_test_avg_spd.shape[0]/ test_avg_spd.shape[0]* 100
        self.num_train_data= train_avg_spd.shape[0]
        self.num_test_data= test_avg_spd.shape[0]
        self.train_avg_spd= train_avg_spd
        self.test_avg_spd= test_avg_spd
        print('[i]    %.2f%% of train images excluded.'%(train_delete_ratio))
        print('[i]    %.2f%% of test  images excluded.'%(test_delete_ratio))

    def data_histogram(self):
        dir_avg= os.path.join(self.dir_processed_data, 'avg_spd')
        avg_list= os.listdir(dir_avg)
        dir_histogram= os.path.join(self.dir_processed_data, 'histograms')
        train_avg_spd= np.load(os.path.join(dir_avg, 'train.npy'))
        test_avg_spd= np.load(os.path.join(dir_avg, 'test.npy'))
        normal_spd= np.load(os.path.join(dir_avg, 'normal.npy'))
        if not os.path.exists(dir_histogram):
            os.mkdir(dir_histogram)
        sporty_spd= []
        mild_spd= []
        for train_idx in range(self.num_train_data):
            avg_spd= train_avg_spd[train_idx, 2].astype(np.float32)
            if train_avg_spd[train_idx, 1].astype(np.int8)== 1:
                sporty_spd.append(avg_spd)
            elif train_avg_spd[train_idx, 1].astype(np.int8)== 0:
                mild_spd.append(avg_spd)
        plt.figure()
        ys, xs, patches= plt.hist(x= sporty_spd, 
                                  bins= 50, density=True,
                                  fc= (1, 0, 0, 0.5))
        plt.title('green:mild,  blue:normal,  red:sporty')
        plt.xlim(-1, 4.5)
        plt.ylim(0, 1)
        ys, xs, patches= plt.hist(x= mild_spd,
                                  bins= 50, density= True, color= 'r',
                                  fc= (0, 1, 0, 0.5))
        ys, xs, patches= plt.hist(x= normal_spd[:, 2].astype(np.float32),
                                  bins= 50, density= True,
                                  fc= (0, 0, 1, 0.5))

        plt.savefig(os.path.join(dir_histogram, 'train.png'))
        

    # sporty, mild의 실제 물리적인 차이점들 분석할 것
