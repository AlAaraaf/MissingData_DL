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
from utils.utils import rmse_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type = int, required=True)
    parser.add_argument("-dataset", type = str, required = True)
    parser.add_argument("-mr", type = float, required=True) # missing rate
    parser.add_argument("-size", type = int, required=True) # sample size
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    num_imputations = 10

    DAE_parameters = {'learning_rate': 0.001,
                        'batch_size': 512,
                        'num_steps_phase1': 200,
                        'num_steps_phase2': 2,
                        'theta': 7}

    gain_parameters = {'batch_size': 256,
                       'hint_rate': 0.13, # MAR
                       'alpha': 20,
                       'iterations': 100,
                       'dlr':0.005,
                       'glr':0.025,
                       'd_gradstep':2,
                       'g_gradstep':1,
                       'log_name':'gain_sim_m1/tuning/'
                       }

    # Load data
    dataset = args.dataset
    file_name = './data/' + dataset + '.csv'
    model_name = "gain"
    miss_mechanism = "MCAR"
    data_df = pd.read_csv(file_name)
    data_x = data_df.values.astype(np.float32)

    save_path = "../training_data/results/{}/{}_{}_{}/{}".format(dataset, miss_mechanism,args.mr, args.size,model_name)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    if dataset == 'house':
        num_index = list(range(-8, 0))
        cat_index = list(range(-data_df.shape[1], -8))
        # get all possible levels for categorical variable
        all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
        all_levels_dict = dict(zip(data_df.columns[cat_index], all_levels))

    elif dataset == 'boston':
        num_index = list(range(-12, 0))
        cat_index = list(range(-data_df.shape[1], -12))
        # get all possible levels for categorical variable
        all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
        all_levels_dict = dict(zip(data_df.columns[cat_index], all_levels))

    elif dataset == 'credit':
        num_index = list(range(-14, 0))
        cat_index = list(range(-data_df.shape[1], -14))
        # get all possible levels for categorical variable
        all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
        all_levels_dict = dict(zip(data_df.columns[cat_index], all_levels))
    
    elif dataset == 'sim_1_tiny' or dataset == 'sim_2_tiny' or dataset == 'sim_1' or dataset == 'sim_2':
        num_index = list(range(0,0))
        cat_index = list(range(-data_df.shape[1],0))
        # get all possible levels for categorical variable
        all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
        all_levels_dict = dict(zip(data_df.columns[cat_index], all_levels))
    
    elif "sim_m" in dataset:
        num_index = list(range(-1,0))
        cat_index = list(range(-data_df.shape[1], -1))
        # get all possible levels for categorical variable
        all_levels = [np.unique(x) for x in data_x[:, cat_index].T]
        all_levels_dict = dict(zip(data_df.columns[cat_index], all_levels))

    rmse_ls = []
    i = args.id
    file_name = '../training_data/samples/{}/{}_{}_{}/sample_{}.csv'.format(dataset, miss_mechanism, args.mr, args.size, i)  
    data_x_i = np.loadtxt('../training_data/samples/{}/complete_{}_{}/sample_{}.csv'.format(dataset, args.mr, args.size,i), delimiter=",").astype(np.float32)

    miss_data_x = np.loadtxt(file_name, delimiter=",").astype(np.float32)
    data_m = 1 - np.isnan(miss_data_x).astype(np.float32)
    if model_name == "mida":
        imputed_list, loss_list = autoencoder_imputation(miss_data_x, data_m,
                                                            cat_index, num_index,
                                                            all_levels, DAE_parameters, 10)
    if model_name == "gain":
        imputed_list, Gloss_list, Dloss_list = gain(miss_data_x, data_m,
                                                        cat_index, num_index,
                                                        all_levels, gain_parameters, 10)

    for l in range(num_imputations):
        np.savetxt(os.path.join(save_path, "imputed_{}_{}.csv".format(i, l)), imputed_list[l], delimiter=",")
    print("{} done!".format(i))