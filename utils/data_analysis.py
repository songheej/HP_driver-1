import os, pdb
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing.factory_module import factory_class

class data_analysis_class(factory_class):
    def __init__(self, sys_flags, data_flags, train_flags):
        super(data_analysis_class, self).__init__(sys_flags, data_flags, train_flags)
        self.sys_flags= sys_flags
        self.data_flags= data_flags
        self.train_flags= train_flags
        
        self.initialization()
        self.data_histogram()

    def initialization(self):
        def make_dir(dir_path):
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        self.dir_base= self.sys_flags.dir_data_analysis
        self.dir_histogram= os.path.join(self.dir_base, 'histograms')
        dir_label= os.path.join(self.dir_processed_data, 'labels')
        self.train_s_names= np.load(os.path.join(dir_label, 'train_s.npy'))
        self.train_m_names= np.load(os.path.join(dir_label, 'train_m.npy'))
        self.test_s_names= np.load(os.path.join(dir_label, 'test_s.npy'))
        self.test_m_names= np.load(os.path.join(dir_label, 'test_m.npy'))
        self.dir_imgs= os.path.join(self.dir_processed_data, 'imgs')
        make_dir(self.dir_base)
        make_dir(self.dir_histogram)

    def data_histogram(self):
        print('[i]  Data histogram module')
        signal_names= self.data_flags.signal_names
        del signal_names[0:4]
        signal_names.append('WHL_SPD_AVG')
        for driver_name in self.train_s_names:
            avg_values= []
            dir_imgs= os.path.join(
            for signal_name in signal_names:
                signal_idx= self.data_flags.signal_order[signal_name]

        for signal_name in signal_names:
            signal_idx= self.data_flags.signal_order[signal_name]
            for driver_name in self.train_s_names:
                avg_value= []
                dir_imgs= os.path.join(self.dir_imgs, driver_name)
                list_imgs= os.listdir(dir_imgs)
                for img_name in list_imgs:
                    dir_img= os.path.join(dir_imgs, img_name)
                    pdb.set_trace()


        avg_list= os.listdir(self.dir_avg)
        train_avg_spd= np.load(os.path.join(self.dir_avg, 'train.npy'))
        test_avg_spd= np.load(os.path.join(self.dir_avg, 'test.npy'))
        normal_avg_spd= np.load(os.path.join(self.dir_avg, 'normal.npy'))
        num_train_data= train_avg_spd.shape[0]
        num_test_data= test_avg_spd.shape[0]
        num_normal_data= normal_avg_spd.shape[0]
        print('[i]    Num Train data %d, Test data %d, normal data %d will be figured'\
                %(num_train_data, num_test_data, num_normal_data))
        # train histogram
        sporty_spd= []
        mild_spd= []
        normal_spd= []
        for idx in range(num_train_data):
            avg_spd= train_avg_spd[idx, 2].astype(np.float32)
            if train_avg_spd[idx, 1].astype(np.int)== 1:
                sporty_spd.append(avg_spd)
            elif train_avg_spd[idx, 1].astype(np.int)== 0:
                mild_spd.append(avg_spd)
        for idx in range(num_normal_data):
            avg_spd= normal_avg_spd[idx, 2].astype(np.float32)
            normal_spd.append(avg_spd)
        plt.figure()
        plt.hist(x= sporty_spd, bins= 50, density= True, fc= (1, 0, 0, 0.5))
        plt.hist(x= mild_spd, bins= 50, density= True, fc= (0, 1, 0, 0.5))
        plt.title('green: mild, red:sporty')
        plt.xlim(-1, 4.5)
        plt.ylim(0, 1)
        plt.savefig(os.path.join(self.dir_histogram, 'train.png'))
