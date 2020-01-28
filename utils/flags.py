import argparse
import os

def get_sys_flags():
    dir_cwd= os.getcwd()
    dir_data_base= os.path.join('/mnt', 'mnt', 'sda', 'Hyundai', 'wif')

    parser= argparse.ArgumentParser()
    # DIRS
    parser.add_argument('--dir_cwd', default= dir_cwd)
    parser.add_argument('--dir_data_base', default= dir_data_base)
    parser.add_argument('--dir_mat', default= os.path.join(dir_data_base, 'mat'))
    parser.add_argument('--dir_annotation', default= os.path.join(dir_data_base, 'annotations'))
    parser.add_argument('--dir_video', default= os.path.join(dir_data_base, 'video'))
    parser.add_argument('--dir_processed_data', default= os.path.join(dir_data_base, 'processed'))
    parser.add_argument('--dir_data_analysis', default= os.path.join('/mnt', 'mnt', 'sdb', 'ysshin',
                                                                     'Hyundai', 'data_analysis'))
    # Modules
    parser.add_argument('--module_factory', default= 'utils.preprocessing.factory_module')
    parser.add_argument('--module_train', default= 'utils.cnn_basic')
    parser.add_argument('--module_test', default= 'utils.test_cnn_basic')
    parser.add_argument('--module_model', default= 'models.cnn_basic_models')
    parser.add_argument('--module_data_analysis', default= 'utils.data_analysis')
    flags= parser.parse_args()
    return flags

def get_data_flags():
    parser= argparse.ArgumentParser()
    parser.add_argument('--signal_names', default= ['WHL_SPD_FL', 'WHL_SPD_FR', 
                                                    'WHL_SPD_RL', 'WHL_SPD_RR',
                                                    'SAS_Speed', 'SAS_Angle', 'LAT_ACCEL',
                                                    'LONG_ACCEL', 'YAW_RATE',
                                                    'TPS', 'CYL_PRES'])
    parser.add_argument('--signal_order', default= {'WHL_SPD_AVG': 0, 'SAS_Speed': 1,
                                                    'SAS_Angle': 2, 'LAT_ACCEL': 3,
                                                    'LONG_ACCEL': 4, 'YAW_RATE': 5,
                                                    'TPS': 6, 'CYL_PRES':7})
    parser.add_argument('--Ts', default= 0.1)
    parser.add_argument('--use_normalize', default= True)
    parser.add_argument('--normalize_value', default= 1)
    parser.add_argument('--Ls', default= 256)
    parser.add_argument('--Ls_shift', default= 10)
    flags= parser.parse_args()
    return flags

def get_train_flags():
    dir_results= os.path.join('/mnt', 'mnt', 'sdb', 'ysshin', 'Hyundai', 'results')
    dir_log= os.path.join('/mnt', 'mnt', 'sdb', 'ysshin', 'Hyundai', 'results', 'train_info.txt')
    parser= argparse.ArgumentParser()
    parser.add_argument('--dir_results', default= dir_results)
    parser.add_argument('--gpu_num', default= 0)
    parser.add_argument('--random_seed_np', default= 777)
    parser.add_argument('--random_seed_tf', default= 777)
    parser.add_argument('--lower_threshold', default= 25)
    parser.add_argument('--upper_threshold', default= 35)
    parser.add_argument('--train_ratio', default= 0.8)
    parser.add_argument('--num_class', default= 2)
    parser.add_argument('--spd_threshold', default= 20)
    parser.add_argument('--model_name', default= 'conv4_fc2')
    parser.add_argument('--use_ma', default= False)

    parser.add_argument('--l2_regul', default= 0.01)
    parser.add_argument('--act_fn', default= 'relu')
    parser.add_argument('--pool_size', default= (1, 2))
    parser.add_argument('--pool_padding', default= 'same')
    parser.add_argument('--conv_padding', default= 'same')
    parser.add_argument('--conv_strides', default= (1,1))

    parser.add_argument('--filters1', default= 64)
    parser.add_argument('--filters2', default= 64)
    parser.add_argument('--filters3', default= 128)
    parser.add_argument('--filters4', default= 128)
    parser.add_argument('--kernel_size1', default= (8, 5))
    parser.add_argument('--kernel_size2', default= (1, 3))
    parser.add_argument('--kernel_size3', default= (1, 3))
    parser.add_argument('--kernel_size4', default= (1, 3))
    parser.add_argument('--fc_unit1', default= 512)

    parser.add_argument('--learning_rate', default= 1e-5)
    parser.add_argument('--batch_size', default= 64)
    parser.add_argument('--test_batch_size', default= 256)
    parser.add_argument('--num_epoch', default= 50)


    flags= parser.parse_args()
    return flags

