from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import argparse

from utils.utils import sample_batch_index, binary_sampler, response_sampler
from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type = str, required=True)
    parser.add_argument("-num_samp", type = str, required=True) # number of samples
    parser.add_argument("-mr", type = float, required=True) # missing rate
    parser.add_argument("-size", type = int, required=True) # sample size
    parser.add_argument("-seed", type = int, required=False, default = 42) # set seed
    parser.add_argument("-missc", type = int, required=False, default=-1) # specify whether only impute on one variable (for simulation)
    return parser.parse_args()

## For missc parameter, if missc == -1, all columns will have missing lines.
## Otherwise, missc will specify a 0-based column index for generating missing lines.

if __name__ == '__main__':
    # Load data
    args = parse_args()
    dataset = args.data

    file_name = '../training_data/origin/' + dataset + '.csv'
    data_df = pd.read_csv(file_name)
    no, dim = data_df.shape

    data_x = data_df.values.astype(np.float32)
    num_samples = int(args.num_samp)
    miss_rate = float(args.mr)
    sample_size = int(args.size)

    save_path_complete = "../training_data/samples/{}/complete_{}_{}".format(dataset,miss_rate, sample_size)
    if not os.path.exists(save_path_complete):
            os.makedirs(save_path_complete)
    
    save_path_mcar = "../training_data/samples/{}/MCAR_{}_{}".format(dataset,miss_rate, sample_size)
    if not os.path.exists(save_path_mcar):
            os.makedirs(save_path_mcar)

    for i in trange(num_samples):
        # random samples
        sample_idx = sample_batch_index(no, sample_size, args.seed + i)
        data_x_i = data_x[sample_idx, :]
        no_i, dim_i = data_x_i.shape
        save_path = "../training_data/samples/{}/complete_{}_{}/sample_{}.csv".format(dataset,miss_rate, sample_size, i)
        np.savetxt(save_path, data_x_i, delimiter=",")

        # Introduce missing data
        if args.missc == -1:
            # missing value across all variables
            data_m = binary_sampler(1 - miss_rate, no_i, dim_i, args.seed + i)
            miss_data_x = data_x_i.copy()
            miss_data_x[data_m == 0] = np.nan
        else:
            # missing value at single variable
            data_m = response_sampler(1 - miss_rate, no_i, dim_i, args.missc, args.seed + i)
            miss_data_x = data_x_i.copy()
            miss_data_x[data_m == 0] = np.nan
        
        # Save files
        save_path = "../training_data/samples/{}/MCAR_{}_{}/sample_{}.csv".format(dataset,miss_rate, sample_size, i)
        np.savetxt(save_path, miss_data_x, delimiter=",")


