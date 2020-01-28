import os, importlib, pdb
os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'
import utils.flags as flags_module
import tensorflow as tf

if __name__== '__main__':
    sys_flags= flags_module.get_sys_flags()
    data_flags= flags_module.get_data_flags()
    train_flags= flags_module.get_train_flags()
    gpus= tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[train_flags.gpu_num], 'GPU')
    importlib.import_module(sys_flags.module_train).train_class(sys_flags, data_flags, train_flags)

    print('[i]  GPU %d/%d is in Use.'%(train_flags.gpu_num, len(gpus)- 1))

