from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pathlib
import os
import argparse

from models.MIDA_v2 import autoencoder_imputation
from models.GAIN_v2 import gain
from models.GAIN_qreg import gain_qrg
from utils.utils import rmse_loss
from process_dataset import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type = int, required=True) # the id of sample used for training
    parser.add_argument("-dataset", type = str, required = True) # the name of dataset
    parser.add_argument("-model", type = str, required = True) # the DNN model used in training
    parser.add_argument("-mr", type = float, required=True) # missing rate
    parser.add_argument("-size", type = int, required=True) # sample size

    # hyper params
    parser.add_argument("-batch_size", type = int, required = True)
    parser.add_argument("-alpha", type = int, required = True) # hyperparam for GAN optim
    parser.add_argument("-iterations", type = int, required = True) # overall number of epochs
    parser.add_argument("-dlr", type = float, required=True) # learning rate for discriminator
    parser.add_argument("-glr", type = float, required = True) # learning rate for generator
    parser.add_argument("-d_gradstep", type=int, required=True) # discriminator steps
    parser.add_argument("-g_gradstep", type=int, required=True) # generator steps
    parser.add_argument("-log_name", type=str, required=True) # the output log name to differ from other experiments
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    num_imputations = 10

    DAE_parameters = {'learning_rate': 0.001,
                        'batch_size': 512,
                        'num_steps_phase1': 200,
                        'num_steps_phase2': 2,
                        'theta': 7}

    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': 0.13, # MAR
                       'alpha': args.alpha,
                       'iterations': args.iterations,
                       'dlr':args.dlr,
                       'glr':args.glr,
                       'd_gradstep':args.d_gradstep,
                       'g_gradstep':args.g_gradstep,
                       'log_name':args.log_name
                       }
    
    gain_qreg_parameters = {'batch_size': args.batch_size,
                       'hint_rate': 0.13, # MAR
                       'alpha': args.alpha,
                       'iterations': args.iterations,
                       'dlr':args.dlr,
                       'glr':args.glr,
                       'd_gradstep':args.d_gradstep,
                       'g_gradstep':args.g_gradstep,
                       'log_name':args.log_name
                       }

    # Load data
    dataset = args.dataset
    model_name = args.model
    miss_mechanism = "MCAR"

    save_path = "../training_data/results/{}/{}_{}_{}/{}".format(dataset, miss_mechanism,args.mr, args.size,model_name)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    rmse_ls = []

    data_x_i, miss_data_x, data_m, all_levels_dict = load_dataset(dataset, miss_mechanism, args.mr, args.size, args.id)
    cat_index = all_levels_dict['cat_index']
    num_index = all_levels_dict['num_index']
    all_levels = list(all_levels_dict['levels'].values())

    if model_name == "mida":
        imputed_list, loss_list = autoencoder_imputation(miss_data_x, data_m,
                                                            cat_index, num_index,
                                                            all_levels, DAE_parameters, 10)
    if model_name == "gain_qreg":
        imputed_list, Gloss_list, Dloss_list = gain_qrg(miss_data_x, data_m,
                                                        cat_index, num_index,
                                                        all_levels, gain_qreg_parameters, 10)
    
    if model_name == "gain":
        imputed_list, Gloss_list, Dloss_list = gain(miss_data_x, data_m,
                                                        cat_index, num_index,
                                                        all_levels, gain_parameters, 10)
    

    for l in range(num_imputations):
        print(rmse_loss(data_x_i, imputed_list[l], data_m))
        np.savetxt(os.path.join(save_path, "imputed_{}_{}.csv".format(args.id, l)), imputed_list[l], delimiter=",")
    print("{} done!".format(args.id))