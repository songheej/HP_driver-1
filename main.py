import argparse
import os
import importlib
import tensorflow as tf
import pdb
import numpy as np

def get_flags():
    dir_cwd= os.getcwd()
    dir_data_base= os.path.join('/mnt', 'mnt', 'sda', 'Hyundai', 'wif')
    dir_annotations= os.path.join(dir_data_base, 'annotations')
    dir_mat= os.path.join(dir_data_base, 'mat')
    dir_video= os.path.join(dir_data_base, 'video')
    dir_processed_data= os.path.join(dir_data_base, 'processed')
    
    parser= argparse.ArgumentParser()
    # DIRS & Module DIRS
    parser.add_argument('--dir_cwd', default= dir_cwd)
    parser.add_argument('--dir_data_base', default= dir_data_base)
    parser.add_argument('--dir_mat', default= dir_mat)
    parser.add_argument('--dir_annotations', default= dir_annotations)
    parser.add_argument('--dir_video', default= dir_video)
    parser.add_argument('--dir_processed_data', default= dir_processed_data)
    parser.add_argument('--factory_module', default= 'utils.factory_module')
    parser.add_argument('--train_module', default= 'train_utils.cnn_basic_module')
    parser.add_argument('--model_module', default= 'models.cnn_basic_models')

    # Data Configuration
    parser.add_argument('--upper_threshold', default= 35)
    parser.add_argument('--lower_threshold', default= 25)
    parser.add_argument('--c_signal_names', default= None)
    parser.add_argument('--p_signal_names', default= None)
    parser.add_argument('--me_signal_names', default= None)
    parser.add_argument('--gps_signal_names', default= None)
    parser.add_argument('--num_signals', default= None)
    parser.add_argument('--Ts', default=0.1)
    parser.add_argument('--use_normalize', default= False)
    parser.add_argument('--normalize_value', default= None)
    parser.add_argument('--use_mean_subtract', default= True)
    parser.add_argument('--Ls', default= None)
    parser.add_argument('--Ls_shift', default= None)
    parser.add_argument('--train_ratio', default= 0.8)
    parser.add_argument('--num_classes', default= 2)
    parser.add_argument('--label_version', default= 'version2')
    # version2: 2개의 binary classifiation
    # version3: 2개 binary 중간도 classificadtion.
    parser.add_argument('--random_seed', default= 777)
    parser.add_argument('--use_moving_avg', default= True)
    parser.add_argument('--moving_avg_window', default= 5)
    parser.add_argument('--use_spd_filtering', default= True)
    parser.add_argument('--spd_threshold', default= 15)

    # Model Configurations
    parser.add_argument('--model_name', default= None)
    # conv3_fc2,  conv5_fc2, conv6_fc3
    parser.add_argument('--last_act_fn', default= 'softmax')

    # Train Configurations
    parser.add_argument('--batch_size', default=None)
    parser.add_argument('--test_batch_size', default=128)
    parser.add_argument('--iterations', default=20000)
    parser.add_argument('--num_epochs', default=100)
    parser.add_argument('--learning_rate', default=0.00001)
    parser.add_argument('--l2_regul', default=None)
    parser.add_argument('--gpu_num', default=0)

    # Model Configurations
    # Conv Layers
    parser.add_argument('--filters1', default=64)
    parser.add_argument('--filters2', default=128)
    parser.add_argument('--filters3', default=256)
    parser.add_argument('--filters4', default=512)
    parser.add_argument('--filters5', default=1024)
    parser.add_argument('--kernel_size1', default=5)  # 다른 하나는 run_time num_signals
    parser.add_argument('--kernel_size2', default=(1,3))
    parser.add_argument('--kernel_size3', default=(1,3))
    parser.add_argument('--kernel_size4', default=(1,3))
    parser.add_argument('--kernel_size5', default=(1,3))
    parser.add_argument('--kernel_size6', default=(1,3))
    parser.add_argument('--act_fn1', default='relu') 
    parser.add_argument('--act_fn2', default='relu')
    parser.add_argument('--act_fn3', default='relu')
    parser.add_argument('--act_fn4', default='relu')
    parser.add_argument('--act_fn5', default='relu')
    parser.add_argument('--act_fn6', default='relu')
    parser.add_argument('--pool_size', default=(1, 2))
    parser.add_argument('--pool_padding', default= 'same')
    parser.add_argument('--embedding_size', default= None)
    parser.add_argument('--first_kernel_size', default= (8, 5))
    parser.add_argument('--kernel_size', default= (1, 5))
    parser.add_argument('--conv_padding', default= 'same')
    parser.add_argument('--conv_strides', default= (1,1))
    parser.add_argument('--use_batchnorm', default= False)
    # FC Layers
    parser.add_argument('--units1', default= 512)
    parser.add_argument('--units2', default= 1024)
    parser.add_argument('--fc_act_fn1', default= 'relu') ####### 'sigmoid'
    parser.add_argument('--fc_act_fn2', default='relu')
    # Dropout
    parser.add_argument('--use_dropout', default= False)

    # Sampling Strategy
    parser.add_argument('--use_ss', default= False)
    parser.add_argument('--ss_batch', default= 512)

    parser.add_argument('--exp_num', default=1)
    
    flags= parser.parse_args()
    # Experiments
    return flags

if __name__== '__main__':
    flags= get_flags()
    flags.c_signal_names= ['WHL_SPD_FL', 'WHL_SPD_FR', 'WHL_SPD_RL', 'WHL_SPD_RR',
                           'SAS_Speed', 'SAS_Angle', 'LAT_ACCEL', 'LONG_ACCEL',
                           'YAW_RATE', 'TPS']
    flags.p_signal_names= ['CYL_PRES']
    flags.me_signal_names= []
    flags.gps_signal_names= []

    flags.gpu_num= 1
    
    flags.Ls= 512
    flags.Ls_shift= 10
    flags.num_signals= len(flags.c_signal_names)+ len(flags.p_signal_names)+\
                       len(flags.me_signal_names)+ len(flags.gps_signal_names)
    flags.use_normalize= True
    flags.normalize_value= 1
    flags.model_name= 'conv4_fc2'
    flags.batch_size= 64
    flags.l2_regul= 0.1
    flags.last_act_fn= 'softmax'

    gpus= tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[flags.gpu_num], 'GPU')
    np.random.seed(777)
    os.environ["TF_CPPMIN_LOG_LEVEL"]='3'
    train_class= importlib.import_module(flags.train_module).train_class(flags, training= True)
