import os
import argparse
import numpy as np
import pathlib
from eval.calculate_metrics import generate_cond_cont, metric_comparison

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", type = int, required=True) # the id of sample used for training
    parser.add_argument("-dataset", type = str, required = True) # the name of dataset
    parser.add_argument("-model", type = str, required = True) # the model used in training
    parser.add_argument("-mr", type = float, required=True) # missing rate
    parser.add_argument("-size", type = int, required=True) # sample size

    parser.add_argument("-batch_size", type = int, required = True)
    parser.add_argument("-alpha", type = int, required = True) # hyperparam for GAN optim
    parser.add_argument("-iterations", type = int, required = True) # overall number of epochs
    parser.add_argument("-dlr", type = float, required=True) # learning rate for discriminator
    parser.add_argument("-glr", type = float, required = True) # learning rate for generator
    parser.add_argument("-d_gradstep", type=int, required=True) # discriminator steps
    parser.add_argument("-g_gradstep", type=int, required=True) # generator steps
    parser.add_argument("-onlylog", type=int, default=0) # generate file prefix only by prefix
    parser.add_argument("-prefix", type=str, required=True) # the prefix name to differ from other experiments

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    mr = args.mr
    size = args.size
    sample_id = args.id
    impute_num = 10
    attn_var = range(1)
    y_loc = 1

    print('Calculating metrics......')
    all_levels, all_levels_comb, cond_dist_complete, perm_data, condtag = generate_cond_cont(dataset, mr, size, sample_id, attn_var, y_loc)

    # calculate metrics
    method_list = [args.model]
    quantile_mse = dict.fromkeys(method_list, None)
    quantile_mae = dict.fromkeys(method_list, None)
    comparison_dict = dict.fromkeys(method_list, None)
    cond_dist_imputed = dict.fromkeys(method_list, None)
    quant_list = [0.1,0.3,0.5,0.7,0.9,1]

    for method in method_list:
        imputed_data_folder = '../training_data/results/' + dataset + '/MCAR_' + str(mr) + '_' + str(size) + '/' + method + '/'
        result = metric_comparison(method,imputed_data_folder, quant_list,
                            all_levels, all_levels_comb, cond_dist_complete, attn_var, condtag,
                            sample_id, impute_num)
        cond_dist_imputed[method], quantile_mse[method], quantile_mae[method] = result
    
    save_path = '../metrics/' + dataset + '_MCAR_' + str(mr) + '_' + str(size) + '/' + method + '/'
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    if args.onlylog == 0:
        hyperparam = str(args.batch_size) + '_' + str(args.alpha) + '_' + str(args.dlr) + '_' + str(args.glr) + '_' + str(args.d_gradstep) + '_' + str(args.g_gradstep) + '_'
    else:
        hyperparam = args.prefix
    maefile = hyperparam + 'quantile_mae.npy'
    msefile = hyperparam + 'quantile_mse.npy'
    np.save(os.path.join(save_path, maefile), quantile_mae)
    np.save(os.path.join(save_path, maefile), quantile_mse)

    print("finish")
    # delete imputed data
