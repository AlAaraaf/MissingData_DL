import os
import shutil
import argparse
import numpy as np
import pandas as pd
import pathlib
from eval.ind_metrics import get_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type = int, required=True) # the id of sample used for training
    parser.add_argument("-dataset", type = str, required = True) # the name of dataset
    parser.add_argument("-model", type = str, required = True) # the model used in training
    parser.add_argument("-mr", type = float, required=True) # missing rate
    parser.add_argument("-size", type = int, required=True) # sample size
    parser.add_argument("-type", type=str, default='ind') # type of evaluation (correlating to the type of missing, choose from ind and mcar)

    parser.add_argument("-batch_size", type = int, required = True)
    parser.add_argument("-alpha", type = int, required = True) # hyperparam for GAN optim
    parser.add_argument("-iterations", type = int, required = True) # overall number of epochs
    parser.add_argument("-dlr", type = float, required=True) # learning rate for discriminator
    parser.add_argument("-glr", type = float, required = True) # learning rate for generator
    parser.add_argument("-d_gradstep", type=int, required=True) # discriminator steps
    parser.add_argument("-g_gradstep", type=int, required=True) # generator steps
    parser.add_argument("-onlylog", type=int, default=0) # generate file prefix only by prefix
    parser.add_argument("-prefix", type=str) # the prefix name to differ from other experiments
    parser.add_argument("-delete_impute",type=int, default=0) # whether delete the imputed data after analysis

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    mr = args.mr
    size = args.size
    sample_id = args.id
    impute_num = 10
    model = args.model

    print('Calculating metrics......')

    complete_data_path = '../training_data/samples/{}/complete_{}_{}/sample_{}.csv'.format(dataset, mr, size, sample_id)
    sample_data_path = '../training_data/samples/{}/MCAR_{}_{}/sample_{}.csv'.format(dataset, mr, size, sample_id)
    
    complete_data = np.loadtxt(complete_data_path, delimiter=",").astype(np.float32)
    sample_data = np.loadtxt(sample_data_path, delimiter=",").astype(np.float32)

    mask_data = np.isnan(sample_data).astype(np.float32)
    
    imputed_data_folder = '../training_data/results/{}/MCAR_{}_{}/{}/'.format(dataset, mr, size, model)
    data_level = np.load('./datalevel.npy', allow_pickle=True).item()[dataset]

    metrics = get_metrics(complete_data, mask_data, imputed_data_folder, data_level, impute_num, sample_id)
    save_path = '../metrics/{}/{}/MCAR_{}_{}/{}/'.format(args.type, dataset, mr, size, model)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    if args.onlylog == 0:
        hyperparam = '{}_{}_{}_{}_{}_{}.npy'.format(str(args.batch_size), str(args.alpha), str(args.dlr), str(args.glr), str(args.d_gradstep), str(args.g_gradstep))
    else:
        hyperparam = args.prefix
    save_name = os.path.join(save_path, hyperparam)
    np.save(save_name, metrics, allow_pickle=True)
    print("Finish.")

    # delete imputed data
    if args.delete_impute == 1:
        shutil.rmtree(imputed_data_folder)
        print("Imputed data deleted.")