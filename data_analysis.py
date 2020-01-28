import numpy as np
import os, pdb
import matplotlib.pyplot as plt
import importlib
import utils.flags as flags_module

if __name__== '__main__':
    sys_flags= flags_module.get_sys_flags()
    data_flags= flags_module.get_data_flags()
    train_flags= flags_module.get_train_flags()
    importlib.import_module(sys_flags.module_data_analysis).data_analysis_class(sys_flags, data_flags, train_flags)
