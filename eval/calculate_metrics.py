import os
import numpy as np
import pandas as pd
from itertools import product
import pathlib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.spatial.distance import jensenshannon

def strint(x):
    return str(int(x))

def generate_cond_cont(data, attn_var, y_loc):

    # get all possible levels' combination for categorical variable that we specified
    all_levels = [np.unique(data.iloc[:,i]).tolist() for i in attn_var]
    all_levels_comb = list(product(*all_levels))

    # extract conditional distributions for complete data
    cond_dist_complete = dict.fromkeys(all_levels_comb, None)
    condtag = []
    
    for index, item in data.iterrows():
        cond = '-'.join([strint(item[i]) for i in attn_var])
        condtag.append(cond)
    
    data['cond'] = condtag
    
    for key in all_levels_comb:
        str_key = '-'.join([strint(key[i]) for i in range(len(key))])
        cond_dist_complete[key] = data.loc[data['cond'] == str_key][y_loc].values.tolist()
    
    pdata = data.copy()
    pdata[y_loc] = np.random.permutation(pdata[y_loc])

    return all_levels, all_levels_comb, cond_dist_complete, pdata, condtag

# output MSE for each pair of specified conditional distribution in one sample
# extract conditional dataset from imputed datasets
def metric_comparison(method, imputed_data_folder, quant_list,
                      all_levels, all_levels_comb, cond_dist_complete, attn_var, condtag,
                      sample_id, impute_num):

    cond_dist_imputed = {key:[] for key in all_levels_comb}
    mse_list = {key:[] for key in all_levels_comb}
    mae_list = {key:[] for key in all_levels_comb}

    for i in range(impute_num):
        current_imputed_dir = imputed_data_folder + 'imputed_' + str(sample_id) + '_' + str(i) + '.csv'
        imputed_data = pd.read_csv(current_imputed_dir, header=None)        
        imputed_data['cond'] = condtag

        for key in all_levels_comb:
            str_key = '-'.join([strint(key[i]) for i in range(len(key))])
            cond_imputed_data = imputed_data.loc[imputed_data['cond'] == str_key][5].values.tolist()
            
            # add data to imputed conditional distribution
            cond_dist_imputed[key] = cond_dist_imputed[key] + cond_imputed_data

            # calculate metrics
            y_true = cond_dist_complete[key]
            mse_list[key].append(mean_squared_error(y_true, cond_imputed_data))
            mae_list[key].append(mean_absolute_error(y_true, cond_imputed_data))

    # calculate average
    mse_quantile = dict.fromkeys(all_levels_comb, None)
    mae_quantile = dict.fromkeys(all_levels_comb, None)
    for key in all_levels_comb:
        mse_quantile[key] = np.quantile(mse_list[key], quant_list)
        mae_quantile[key] = np.quantile(mae_list[key], quant_list)
        
    return cond_dist_imputed, mse_quantile, mae_quantile


# output estimated JS divergence for each pair of specified conditional distribution in one sample
# extract conditional dataset from imputed datasets
def js_comparison(imputed_data_folder,dataset,
                      all_levels_comb, cond_dist_complete, condtag,
                      sample_id, impute_num, y_loc):
    
    all_levels_dict = np.load('datalevel.npy', allow_pickle='TRUE').item()[dataset]
    y_levels = list(all_levels_dict['levels'].values())[y_loc] 
    cond_dist_imputed = {key:[] for key in all_levels_comb}
    js_val = {key:[] for key in all_levels_comb}

    for i in range(impute_num):
        current_imputed_dir = imputed_data_folder + 'imputed_' + str(sample_id) + '_' + str(i) + '.csv'
        imputed_data = pd.read_csv(current_imputed_dir, header=None)        
        imputed_data['cond'] = condtag

        for key in all_levels_comb:
            str_key = '-'.join([strint(key[i]) for i in range(len(key))])
            cond_imputed_data = [x for x in imputed_data.loc[imputed_data['cond'] == str_key][y_loc].values.tolist()]
            
            # add data to imputed conditional distribution
            cond_dist_imputed[key] = cond_dist_imputed[key] + cond_imputed_data

    def get_prob(labels):
        test = pd.Series([i for i in labels])
        counts = []
        for i in range(len(y_levels)):
            counts.append(sum(x == y_levels[i] for x in labels))
        counts = [x / len(test) for x in counts]
        return counts
    
    for key in all_levels_comb:
        y_true = get_prob(cond_dist_complete[key])
        y_pred = get_prob(cond_dist_imputed[key])
        # calculate metrics
        js_val[key].append(round(jensenshannon(p = y_pred,q = y_true),6))
        
    return cond_dist_imputed, js_val


def conditional_metrics(args):
    dataset = args.dataset
    mr = args.mr
    size = args.size
    sample_id = args.id
    impute_num = 10
    attn_var = range(1)
    y_loc = 1

    
    complete_data_path = '../training_data/samples/{}/complete_{}_{}/sample_{}.csv'.format(dataset, mr, size, sample_id)
    complete_data = pd.read_csv(complete_data_path, header=None)
    all_levels, all_levels_comb, cond_dist_complete, perm_data, condtag = generate_cond_cont(complete_data, attn_var, y_loc)

    # calculate metrics
    method_list = [args.model]
    quantile_mse = dict.fromkeys(method_list, None)
    quantile_mae = dict.fromkeys(method_list, None)
    comparison_dict = dict.fromkeys(method_list, None)
    cond_dist_imputed = dict.fromkeys(method_list, None)
    cond_dist_imputed_js = dict.fromkeys(method_list, None)
    js_val =dict.fromkeys(method_list, None)
    quant_list = [0.1,0.3,0.5,0.7,0.9,1]

    for method in method_list:
        imputed_data_folder = '../training_data/results/' + dataset + '/MCAR_' + str(mr) + '_' + str(size) + '/' + method + '/'
        result = metric_comparison(method,imputed_data_folder, quant_list,
                            all_levels, all_levels_comb, cond_dist_complete, attn_var, condtag,
                            sample_id, impute_num)
        cond_dist_imputed[method], quantile_mse[method], quantile_mae[method] = result

        result = js_comparison(imputed_data_folder, dataset,
                         all_levels_comb, cond_dist_complete, condtag,
                         sample_id, impute_num, y_loc)
        cond_dist_imputed_js[method], js_val[method] = result
    
    save_path = '../metrics/' + dataset + '_MCAR_' + str(mr) + '_' + str(size) + '/' + method + '/'
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    
    if args.onlylog == 0:
        hyperparam = str(args.batch_size) + '_' + str(args.alpha) + '_' + str(args.dlr) + '_' + str(args.glr) + '_' + str(args.d_gradstep) + '_' + str(args.g_gradstep) + '_'
    else:
        hyperparam = args.prefix
    maefile = hyperparam + 'quantile_mae.npy'
    msefile = hyperparam + 'quantile_mse.npy'
    jsdivfile = hyperparam + 'jsdiv.npy'
    np.save(os.path.join(save_path, maefile), quantile_mae)
    np.save(os.path.join(save_path, msefile), quantile_mse)
    np.save(os.path.join(save_path, jsdivfile), js_val)

    print("finish")