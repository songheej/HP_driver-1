import os
import numpy as np
import pandas as pd
import pickle
import pdb

class dir_setting:
    def __init__(self, sys_flags, data_flags, train_flags):
        self.sys_flags= sys_flags
        self.data_flags= data_flags
        self.train_flags= train_flags
        self.get_driver_names()
        self.delete_driver_names()
        self.get_annotations()
        self.binary_labeling()
        self.prepare_dirs()

    def get_driver_names(self):
        raw_driver_names= os.listdir(self.sys_flags.dir_mat)
        driver_names= []
        for name in raw_driver_names:
            filtered= name.split('.')[0]                 # Extension out
            filtered= '_'.join(filtered.split('_')[1:])  # Recorder out
            filtered= '-'.join(filtered.split('-')[1:])  # Year out
            driver_names.append(filtered)
        self.driver_names= np.array(driver_names)
        self.driver_names.sort()
        self.num_drivers= len(self.driver_names)

    def delete_driver_names(self):
        delete_names= ['09-26_12-53-30', '10-01_15-43-32', '10-02_15-28-41']
        if delete_names:
            for name in delete_names:
                delete_idx= np.where(self.driver_names== name)[0]
                self.driver_names= np.delete(self.driver_names, obj=delete_idx)
            self.num_drivers= len(self.driver_names)

    def get_annotations(self):
        sys_flags= self.sys_flags
        rater= pd.read_excel(os.path.join(sys_flags.dir_annotation, 'rater.xlsx'))
        before_driving= pd.read_excel(os.path.join(sys_flags.dir_annotation, 'before_driving.xlsx'))
        after_driving= pd.read_excel(os.path.join(sys_flags.dir_annotation, 'after_driving.xlsx'))

        rater_useful_key_num= 8
        self.rater_values= np.zeros(shape= (self.num_drivers, rater_useful_key_num))
        rater_names= rater['date'].values
        label_idx= 0
        for driver_idx, driver_name in enumerate(self.driver_names):
            rater_idx= np.where(driver_name== rater_names)[0]
            for label_idx in range(1, 9):
                col_name= rater.keys()[label_idx]
                self.rater_values[driver_idx, label_idx- 1]= rater[col_name][rater_idx]
    
    def binary_labeling(self):
        label_sum= np.sum(self.rater_values, axis=1)
        self.mild_names= []
        self.mid_names= []
        self.sporty_names= []
        self.int_labels= np.ones(shape= self.num_drivers)*(-2)
        self.int_labels= np.zeros(shape= self.num_drivers)
        for driver_idx, driver_name in enumerate(self.driver_names):
            sum_value= label_sum[driver_idx]
            if sum_value< self.train_flags.lower_threshold:
                self.int_labels[driver_idx]= 0
                self.mild_names.append(driver_name)
            elif sum_value> self.train_flags.upper_threshold:
                self.int_labels[driver_idx]= 1
                self.sporty_names.append(driver_name)
            else:
                self.int_labels[driver_idx]= -1
                self.mid_names.append(driver_name)
        self.mild_names= np.array(self.mild_names)
        self.mid_names= np.array(self.mid_names)
        self.sporty_names= np.array(self.sporty_names)
        self.driver_names_threshold= np.append(self.mild_names, self.sporty_names)
        self.num_drivers_threshold= len(self.driver_names_threshold)
        print('[i]  Binary Labeling Completed.')
        print('[i]    Num drivers changed %d to %d'%(self.num_drivers, self.num_drivers_threshold))
        print('[i]    Mild %d,  Sporty %d'%(len(self.mild_names), len(self.sporty_names)))

    def prepare_dirs(self):
        data_flags_dict= vars(self.data_flags)
        self.dir_processed_data= self.sys_flags.dir_processed_data
        if not os.path.exists(self.dir_processed_data):
            os.mkdir(self.dir_processed_data)
        env_list= os.listdir(self.dir_processed_data)
        if not env_list:
            self.dir_processed_data= os.path.join(self.dir_processed_data, 'env_1')
            os.mkdir(self.dir_processed_data)
            with open(os.path.join(self.dir_processed_data, 'data_flags.pkl'), 'wb') as f:
                pickle.dump(data_flags_dict, f, pickle.HIGHEST_PROTOCOL)
        elif env_list:
            find_same_data_flag= False
            for env_name in env_list:
                dir_env_pkl= os.path.join(self.dir_processed_data, env_name, 'data_flags.pkl')
                with open(dir_env_pkl, 'rb') as f:
                    env_pkl= pickle.load(f)
                if len(data_flags_dict)!= len(env_pkl):
                    pass
                elif data_flags_dict== env_pkl:
                    self.dir_processed_data= os.path.join(self.dir_processed_data, env_name)
                    find_same_data_flag= True
                    break
            if find_same_data_flag== False:
                new_env_name= 'env_%d'%(len(env_list)+ 1)
                self.dir_processed_data= os.path.join(self.dir_processed_data, new_env_name)
                os.mkdir(self.dir_processed_data)
                with open(os.path.join(self.dir_processed_data, 'data_flags.pkl'), 'wb') as f:
                    pickle.dump(data_flags_dict, f, pickle.HIGHEST_PROTOCOL)

